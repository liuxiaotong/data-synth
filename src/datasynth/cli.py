"""DataSynth CLI - 命令行界面."""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click

from datasynth import __version__
from datasynth.config import SynthesisConfig
from datasynth.synthesizer import DataSynthesizer, InteractiveSynthesizer

# Mapping from config file keys to SynthesisConfig fields
_CONFIG_KEYS = {
    "target_count", "model", "provider", "temperature", "diversity_factor",
    "batch_size", "max_retries", "retry_delay", "concurrency", "validate",
    "seed_sample_count", "data_type", "max_tokens_per_sample",
}


def _load_config_file(path: str) -> dict:
    """Load a JSON config file and return valid SynthesisConfig fields."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k in _CONFIG_KEYS}


@click.group()
@click.version_option(version=__version__, prog_name="datasynth")
@click.option("-v", "--verbose", is_flag=True, help="显示详细日志 (DEBUG 级别)")
def main(verbose: bool):
    """DataSynth - 数据合成工具

    基于种子数据和 Schema 批量生成高质量训练数据。
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(name)s %(levelname)s: %(message)s",
    )


@main.command()
@click.argument("analysis_dir", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="输出文件路径 (默认: analysis_dir/11_合成数据/synthetic.json)",
)
@click.option("-n", "--count", type=int, default=100, help="生成数量 (默认: 100)")
@click.option("-m", "--model", type=str, default="claude-sonnet-4-20250514", help="LLM 模型")
@click.option(
    "-p",
    "--provider",
    type=click.Choice(["anthropic", "openai"]),
    default="anthropic",
    help="LLM 提供商",
)
@click.option("-t", "--temperature", type=float, default=0.8, help="采样温度 (0.0-1.0)")
@click.option("--batch-size", type=int, default=5, help="每批生成数量")
@click.option("--max-retries", type=int, default=3, help="失败重试次数 (默认: 3)")
@click.option("--retry-delay", type=float, default=2.0, help="重试间隔秒数 (默认: 2.0)")
@click.option("--concurrency", type=int, default=1, help="并发批次数 (默认: 1)")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "jsonl"]),
    default="json",
    help="输出格式 (默认: json)",
)
@click.option("--dry-run", is_flag=True, help="仅估算成本，不实际生成")
@click.option(
    "--post-hook",
    type=str,
    default=None,
    help="生成成功后执行的命令，支持 {analysis_dir} {output_path} {count} 变量",
)
@click.option(
    "--data-type",
    type=click.Choice(["auto", "instruction_response", "preference", "multi_turn"]),
    default="auto",
    help="数据类型 (auto=自动检测)",
)
@click.option("--resume", is_flag=True, help="增量模式: 从已有输出文件继续生成")
@click.option("--stats", is_flag=True, help="输出生成统计报告")
@click.option("--config", "config_file", type=click.Path(exists=True), help="配置文件 (JSON)")
@click.option("--no-validate", is_flag=True, help="跳过 Schema 验证和去重")
def generate(
    analysis_dir: str,
    output: Optional[str],
    count: int,
    model: str,
    provider: str,
    temperature: float,
    batch_size: int,
    max_retries: int,
    retry_delay: float,
    concurrency: int,
    output_format: str,
    dry_run: bool,
    post_hook: Optional[str],
    data_type: str,
    resume: bool,
    stats: bool,
    config_file: Optional[str],
    no_validate: bool,
):
    """从 DataRecipe 分析结果生成合成数据

    ANALYSIS_DIR: DataRecipe 分析输出目录的路径
    """
    # Load config file as base, explicit CLI flags override
    file_cfg = _load_config_file(config_file) if config_file else {}
    ctx = click.get_current_context()

    def _pick(param_name: str, cli_val, cfg_key: str = ""):
        """Use CLI value if explicitly provided, else config file, else CLI default."""
        src = ctx.get_parameter_source(param_name)
        if src == click.core.ParameterSource.COMMANDLINE:
            return cli_val
        return file_cfg.get(cfg_key or param_name, cli_val)

    count = _pick("count", count, "target_count")
    config = SynthesisConfig(
        target_count=count,
        model=_pick("model", model),
        provider=_pick("provider", provider),
        temperature=_pick("temperature", temperature),
        batch_size=_pick("batch_size", batch_size),
        max_retries=_pick("max_retries", max_retries),
        retry_delay=_pick("retry_delay", retry_delay),
        concurrency=_pick("concurrency", concurrency),
        data_type=_pick("data_type", data_type),
        validate=not no_validate,
    )

    if dry_run:
        estimate = config.estimate_cost(count)
        # Try to load schema for extra info
        schema_path = Path(analysis_dir) / "04_复刻指南" / "DATA_SCHEMA.json"
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_data = json.load(f)
            from datasynth.config import DataSchema

            ds = DataSchema.from_dict(schema_data)
            detected = ds.detect_data_type()
            effective_type = data_type if data_type != "auto" else (detected or "通用")
            click.echo("Schema 信息:")
            click.echo(f"  项目: {ds.project_name or '(未指定)'}")
            click.echo(f"  字段: {', '.join(f.name for f in ds.fields)}")
            click.echo(f"  数据类型: {effective_type}")
            guidelines_path = Path(analysis_dir) / "03_标注规范" / "ANNOTATION_SPEC.md"
            if guidelines_path.exists():
                click.echo("  标注规范: ✓ 已加载")
        click.echo("成本估算:")
        click.echo(f"  目标数量: {estimate['target_count']}")
        click.echo(f"  预计批次: {estimate['estimated_batches']}")
        click.echo(
            f"  预计 Token: {estimate['estimated_input_tokens'] + estimate['estimated_output_tokens']:,}"
        )
        click.echo(f"  预计成本: ${estimate['estimated_cost_usd']:.2f}")
        click.echo(f"  模型: {estimate['model']}")
        return

    click.echo(f"正在从 {analysis_dir} 生成合成数据...")
    click.echo(f"  目标数量: {count}")
    click.echo(f"  模型: {model}")
    if resume:
        click.echo("  模式: 增量续跑 (--resume)")

    def on_progress(current, total):
        click.echo(f"  进度: {current}/{total}", nl=False)
        click.echo("\r", nl=False)

    synthesizer = DataSynthesizer(config)
    result = synthesizer.synthesize_from_datarecipe(
        analysis_dir=analysis_dir,
        output_path=output,
        target_count=count,
        on_progress=on_progress,
        output_format=output_format,
        resume=resume,
    )

    click.echo("")  # New line after progress

    if result.success:
        click.echo(f"✓ 生成成功: {result.output_path}")
        click.echo(f"  生成数量: {result.generated_count}")
        click.echo(f"  失败数量: {result.failed_count}")
        if result.dedup_count:
            click.echo(f"  去重数量: {result.dedup_count}")
        click.echo(f"  Token 用量: {result.total_tokens:,}")
        click.echo(f"  预计成本: ${result.estimated_cost:.4f}")
        click.echo(f"  耗时: {result.duration_seconds:.1f}s")

        # Write stats
        if stats and result.stats:
            stats_path = Path(result.output_path).with_suffix(".stats.json")
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(result.stats, f, indent=2, ensure_ascii=False)
            click.echo(f"  统计报告: {stats_path}")

        # Run post-hook
        if post_hook:
            cmd = post_hook.format(
                analysis_dir=analysis_dir,
                output_path=result.output_path,
                count=result.generated_count,
            )
            click.echo(f"  执行 post-hook: {cmd}")
            hook_result = subprocess.run(cmd, shell=True)
            if hook_result.returncode != 0:
                click.echo(f"  ⚠ post-hook 退出码: {hook_result.returncode}", err=True)
    else:
        click.echo(f"✗ 生成失败: {result.error}", err=True)
        sys.exit(1)


@main.command()
@click.argument("schema_file", type=click.Path(exists=True))
@click.argument("seeds_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), required=True, help="输出文件路径")
@click.option("-n", "--count", type=int, default=100, help="生成数量")
@click.option("-m", "--model", type=str, default="claude-sonnet-4-20250514", help="LLM 模型")
@click.option(
    "-p",
    "--provider",
    type=click.Choice(["anthropic", "openai"]),
    default="anthropic",
    help="LLM 提供商",
)
@click.option("-t", "--temperature", type=float, default=0.8, help="采样温度 (0.0-1.0)")
@click.option("--batch-size", type=int, default=5, help="每批生成数量")
@click.option("--max-retries", type=int, default=3, help="失败重试次数 (默认: 3)")
@click.option("--retry-delay", type=float, default=2.0, help="重试间隔秒数 (默认: 2.0)")
@click.option("--concurrency", type=int, default=1, help="并发批次数 (默认: 1)")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "jsonl"]),
    default="json",
    help="输出格式 (默认: json)",
)
@click.option(
    "--data-type",
    type=click.Choice(["auto", "instruction_response", "preference", "multi_turn"]),
    default="auto",
    help="数据类型 (auto=自动检测)",
)
@click.option("--resume", is_flag=True, help="增量模式: 从已有输出文件继续生成")
@click.option("--stats", is_flag=True, help="输出生成统计报告")
@click.option("--dry-run", is_flag=True, help="仅估算成本，不实际生成")
@click.option(
    "--post-hook",
    type=str,
    default=None,
    help="生成成功后执行的命令，支持 {output_path} {count} 变量",
)
@click.option("--config", "config_file", type=click.Path(exists=True), help="配置文件 (JSON)")
@click.option("--no-validate", is_flag=True, help="跳过 Schema 验证和去重")
def create(
    schema_file: str,
    seeds_file: str,
    output: str,
    count: int,
    model: str,
    provider: str,
    temperature: float,
    batch_size: int,
    max_retries: int,
    retry_delay: float,
    concurrency: int,
    output_format: str,
    data_type: str,
    resume: bool,
    stats: bool,
    dry_run: bool,
    post_hook: Optional[str],
    config_file: Optional[str],
    no_validate: bool,
):
    """从 Schema 和种子数据创建合成数据

    SCHEMA_FILE: 数据 Schema JSON 文件
    SEEDS_FILE: 种子数据 JSON 文件
    """
    # Load schema
    with open(schema_file, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Load seeds
    with open(seeds_file, "r", encoding="utf-8") as f:
        seeds_data = json.load(f)

    # Support different formats
    if isinstance(seeds_data, list):
        seeds = seeds_data
    else:
        seeds = seeds_data.get("samples", seeds_data.get("seeds", []))

    # Extract data from samples
    seed_samples = []
    for s in seeds:
        if "data" in s:
            seed_samples.append(s["data"])
        else:
            seed_samples.append(s)

    if not seed_samples:
        click.echo("✗ 种子数据为空", err=True)
        sys.exit(1)

    # Load config file as base, explicit CLI flags override
    file_cfg = _load_config_file(config_file) if config_file else {}
    ctx = click.get_current_context()

    def _pick(param_name: str, cli_val, cfg_key: str = ""):
        src = ctx.get_parameter_source(param_name)
        if src == click.core.ParameterSource.COMMANDLINE:
            return cli_val
        return file_cfg.get(cfg_key or param_name, cli_val)

    count = _pick("count", count, "target_count")
    config = SynthesisConfig(
        target_count=count,
        model=_pick("model", model),
        provider=_pick("provider", provider),
        temperature=_pick("temperature", temperature),
        batch_size=_pick("batch_size", batch_size),
        max_retries=_pick("max_retries", max_retries),
        retry_delay=_pick("retry_delay", retry_delay),
        concurrency=_pick("concurrency", concurrency),
        data_type=_pick("data_type", data_type),
        validate=not no_validate,
    )

    if dry_run:
        estimate = config.estimate_cost(count)
        click.echo("成本估算:")
        click.echo(f"  目标数量: {estimate['target_count']}")
        click.echo(f"  预计批次: {estimate['estimated_batches']}")
        click.echo(
            f"  预计 Token: {estimate['estimated_input_tokens'] + estimate['estimated_output_tokens']:,}"
        )
        click.echo(f"  预计成本: ${estimate['estimated_cost_usd']:.2f}")
        click.echo(f"  模型: {estimate['model']}")
        return

    click.echo("正在生成合成数据...")
    click.echo(f"  Schema: {schema_file}")
    click.echo(f"  种子数量: {len(seed_samples)}")
    click.echo(f"  目标数量: {count}")

    synthesizer = DataSynthesizer(config)
    result = synthesizer.synthesize(
        schema=schema,
        seed_samples=seed_samples,
        output_path=output,
        target_count=count,
        output_format=output_format,
        resume=resume,
    )

    if result.success:
        click.echo(f"✓ 生成成功: {result.output_path}")
        click.echo(f"  生成数量: {result.generated_count}")
        if result.dedup_count:
            click.echo(f"  去重数量: {result.dedup_count}")

        if stats and result.stats:
            stats_path = Path(output).with_suffix(".stats.json")
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(result.stats, f, indent=2, ensure_ascii=False)
            click.echo(f"  统计报告: {stats_path}")

        # Run post-hook
        if post_hook:
            cmd = post_hook.format(
                output_path=result.output_path,
                count=result.generated_count,
            )
            click.echo(f"  执行 post-hook: {cmd}")
            hook_result = subprocess.run(cmd, shell=True)
            if hook_result.returncode != 0:
                click.echo(f"  ⚠ post-hook 退出码: {hook_result.returncode}", err=True)
    else:
        click.echo(f"✗ 生成失败: {result.error}", err=True)
        sys.exit(1)


@main.command()
@click.argument("analysis_dir", type=click.Path(exists=True))
@click.option("-n", "--count", type=int, default=10, help="生成数量")
@click.option("-o", "--output", type=click.Path(), help="Prompt 输出路径")
@click.option(
    "--data-type",
    type=click.Choice(["auto", "instruction_response", "preference", "multi_turn"]),
    default="auto",
    help="数据类型 (auto=自动检测)",
)
def prepare(
    analysis_dir: str,
    count: int,
    output: Optional[str],
    data_type: str,
):
    """准备合成 Prompt (交互模式)

    生成 Prompt 用于手动调用 LLM 或在 Claude Code 中使用。

    ANALYSIS_DIR: DataRecipe 分析输出目录的路径
    """
    analysis_dir = Path(analysis_dir)

    # Load schema
    schema_path = analysis_dir / "04_复刻指南" / "DATA_SCHEMA.json"
    if not schema_path.exists():
        click.echo(f"✗ Schema 未找到: {schema_path}", err=True)
        sys.exit(1)

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Load seeds
    samples_path = analysis_dir / "09_样例数据" / "samples.json"
    if not samples_path.exists():
        click.echo(f"✗ 种子数据未找到: {samples_path}", err=True)
        sys.exit(1)

    with open(samples_path, "r", encoding="utf-8") as f:
        samples_data = json.load(f)

    seed_samples = []
    for s in samples_data.get("samples", []):
        if "data" in s:
            seed_samples.append(s["data"])
        else:
            seed_samples.append(s)

    # Load guidelines if available
    guidelines = None
    guidelines_path = analysis_dir / "03_标注规范" / "ANNOTATION_SPEC.md"
    if guidelines_path.exists():
        guidelines = guidelines_path.read_text(encoding="utf-8")

    synthesizer = InteractiveSynthesizer()
    result = synthesizer.prepare_synthesis(
        schema=schema,
        seed_samples=seed_samples,
        count=count,
        guidelines=guidelines,
        data_type=data_type,
    )

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["prompt"])
        click.echo(f"✓ Prompt 已保存: {output}")
    else:
        click.echo(result["prompt"])

    click.echo(f"\n---\n预期生成 {count} 条数据")


@main.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("schema_file", type=click.Path(exists=True))
def validate(data_file: str, schema_file: str):
    """验证数据文件是否符合 Schema

    DATA_FILE: 数据文件路径 (JSON / JSONL)
    SCHEMA_FILE: Schema JSON 文件路径
    """
    from datasynth.config import DataSchema

    # Load schema
    with open(schema_file, "r", encoding="utf-8") as f:
        schema = DataSchema.from_dict(json.load(f))

    # Load data
    data_path = Path(data_file)
    samples: list = []
    with open(data_path, "r", encoding="utf-8") as f:
        if data_path.suffix == ".jsonl":
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        else:
            data = json.load(f)
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict) and "samples" in data:
                samples = [s.get("data", s) for s in data["samples"]]

    if not samples:
        click.echo("✗ 数据文件为空", err=True)
        sys.exit(1)

    click.echo(f"验证 {len(samples)} 条数据...")
    click.echo(f"  Schema: {schema_file}")
    click.echo(f"  字段: {', '.join(f.name for f in schema.fields)}")

    valid_count = 0
    error_count = 0
    all_errors: list[tuple[int, list[str]]] = []

    for i, sample in enumerate(samples):
        errs = schema.validate_sample(sample)
        if errs:
            error_count += 1
            all_errors.append((i + 1, errs))
        else:
            valid_count += 1

    click.echo(f"\n结果:")
    click.echo(f"  ✓ 合规: {valid_count}")
    click.echo(f"  ✗ 不合规: {error_count}")

    if all_errors:
        click.echo(f"\n错误详情 (前 10 条):")
        for idx, errs in all_errors[:10]:
            click.echo(f"  #{idx}: {'; '.join(errs)}")
        if len(all_errors) > 10:
            click.echo(f"  ... 共 {len(all_errors)} 条错误")
        sys.exit(1)
    else:
        click.echo("  全部通过 ✓")


@main.command()
@click.option("-o", "--output", type=click.Path(), default=".", help="输出目录 (默认: 当前目录)")
def init(output: str):
    """初始化配置和 Schema 模板

    在指定目录生成 datasynth.config.json 和 schema.json 模板。
    """
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "datasynth.config.json"
    schema_path = out_dir / "schema.json"
    seeds_path = out_dir / "seeds.json"

    config_template = {
        "target_count": 100,
        "model": "claude-sonnet-4-20250514",
        "provider": "anthropic",
        "temperature": 0.8,
        "batch_size": 5,
        "max_retries": 3,
        "retry_delay": 2.0,
        "concurrency": 1,
        "data_type": "auto",
    }

    schema_template = {
        "project_name": "我的项目",
        "description": "项目描述",
        "fields": [
            {"name": "instruction", "type": "text", "description": "用户指令"},
            {"name": "response", "type": "text", "description": "模型回答"},
        ],
    }

    seeds_template = [
        {"instruction": "示例问题 1", "response": "示例回答 1"},
        {"instruction": "示例问题 2", "response": "示例回答 2"},
    ]

    created = []
    for path, data in [
        (config_path, config_template),
        (schema_path, schema_template),
        (seeds_path, seeds_template),
    ]:
        if path.exists():
            click.echo(f"  跳过 (已存在): {path}")
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            created.append(str(path))
            click.echo(f"  ✓ 已创建: {path}")

    if created:
        click.echo(f"\n快速开始:")
        click.echo(f"  knowlyr-datasynth create {schema_path} {seeds_path} -o output.json -n 10")
    else:
        click.echo("\n所有文件已存在，未创建新文件。")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), required=True, help="输出文件路径")
def convert(input_file: str, output: str):
    """转换数据格式 (JSON ↔ JSONL)

    根据输出文件扩展名自动选择格式。
    支持输入格式: JSON (list 或 {samples: [...]})、JSONL。

    INPUT_FILE: 输入数据文件路径
    """
    input_path = Path(input_file)
    output_path = Path(output)

    # Load input
    samples: list = []
    with open(input_path, "r", encoding="utf-8") as f:
        if input_path.suffix == ".jsonl":
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        else:
            data = json.load(f)
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict) and "samples" in data:
                samples = [s.get("data", s) for s in data["samples"]]

    if not samples:
        click.echo("✗ 输入文件为空", err=True)
        sys.exit(1)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

    click.echo(f"✓ 转换完成: {input_path} → {output_path}")
    click.echo(f"  数据条数: {len(samples)}")


@main.command()
@click.option("-n", "--count", type=int, default=100, help="目标数量")
@click.option("-m", "--model", type=str, default="claude-sonnet-4-20250514", help="模型")
def estimate(count: int, model: str):
    """估算生成成本"""
    config = SynthesisConfig(target_count=count, model=model)
    estimate = config.estimate_cost()

    click.echo("成本估算:")
    click.echo(f"  目标数量: {estimate['target_count']}")
    click.echo(f"  预计批次: {estimate['estimated_batches']}")
    click.echo(f"  预计输入 Token: {estimate['estimated_input_tokens']:,}")
    click.echo(f"  预计输出 Token: {estimate['estimated_output_tokens']:,}")
    click.echo(f"  预计成本: ${estimate['estimated_cost_usd']:.2f}")
    click.echo(f"  模型: {estimate['model']}")


if __name__ == "__main__":
    main()
