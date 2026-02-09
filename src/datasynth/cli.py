"""DataSynth CLI - 命令行界面."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click

from datasynth import __version__
from datasynth.config import SynthesisConfig
from datasynth.synthesizer import DataSynthesizer, InteractiveSynthesizer


@click.group()
@click.version_option(version=__version__, prog_name="datasynth")
def main():
    """DataSynth - 数据合成工具

    基于种子数据和 Schema 批量生成高质量训练数据。
    """
    pass


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
):
    """从 DataRecipe 分析结果生成合成数据

    ANALYSIS_DIR: DataRecipe 分析输出目录的路径
    """
    config = SynthesisConfig(
        target_count=count,
        model=model,
        provider=provider,
        temperature=temperature,
        batch_size=batch_size,
        max_retries=max_retries,
        retry_delay=retry_delay,
        concurrency=concurrency,
        data_type=data_type,
    )

    if dry_run:
        # Just estimate cost
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

    click.echo("正在生成合成数据...")
    click.echo(f"  Schema: {schema_file}")
    click.echo(f"  种子数量: {len(seed_samples)}")
    click.echo(f"  目标数量: {count}")

    config = SynthesisConfig(
        target_count=count,
        model=model,
        provider=provider,
        temperature=temperature,
        batch_size=batch_size,
        max_retries=max_retries,
        retry_delay=retry_delay,
        concurrency=concurrency,
        data_type=data_type,
    )

    synthesizer = DataSynthesizer(config)
    result = synthesizer.synthesize(
        schema=schema,
        seed_samples=seed_samples,
        output_path=output,
        target_count=count,
        output_format=output_format,
    )

    if result.success:
        click.echo(f"✓ 生成成功: {result.output_path}")
        click.echo(f"  生成数量: {result.generated_count}")
        if result.dedup_count:
            click.echo(f"  去重数量: {result.dedup_count}")
    else:
        click.echo(f"✗ 生成失败: {result.error}", err=True)
        sys.exit(1)


@main.command()
@click.argument("analysis_dir", type=click.Path(exists=True))
@click.option("-n", "--count", type=int, default=10, help="生成数量")
@click.option("-o", "--output", type=click.Path(), help="Prompt 输出路径")
def prepare(
    analysis_dir: str,
    count: int,
    output: Optional[str],
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

    synthesizer = InteractiveSynthesizer()
    result = synthesizer.prepare_synthesis(
        schema=schema,
        seed_samples=seed_samples,
        count=count,
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
