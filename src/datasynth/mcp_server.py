"""DataSynth MCP Server - Model Context Protocol 服务."""

import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from datasynth.config import SynthesisConfig
from datasynth.synthesizer import DataSynthesizer, InteractiveSynthesizer


def create_server() -> "Server":
    """创建 MCP 服务器实例."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install datasynth[mcp]")

    server = Server("datasynth")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """列出可用的工具."""
        return [
            Tool(
                name="prepare_synthesis",
                description="准备数据合成 Prompt（交互模式，不直接调用 LLM）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "analysis_dir": {
                            "type": "string",
                            "description": "DataRecipe 分析输出目录的路径",
                        },
                        "count": {
                            "type": "integer",
                            "description": "要生成的数据条数 (默认: 10)",
                            "default": 10,
                        },
                        "data_type": {
                            "type": "string",
                            "enum": ["auto", "instruction_response", "preference", "multi_turn"],
                            "description": "数据类型 (默认: auto)",
                        },
                    },
                    "required": ["analysis_dir"],
                },
            ),
            Tool(
                name="parse_synthesis_result",
                description="解析 LLM 生成的合成数据并保存",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "response_text": {
                            "type": "string",
                            "description": "LLM 返回的生成结果文本",
                        },
                        "schema_path": {
                            "type": "string",
                            "description": "Schema 文件路径",
                        },
                        "output_path": {
                            "type": "string",
                            "description": "输出文件路径",
                        },
                    },
                    "required": ["response_text", "schema_path", "output_path"],
                },
            ),
            Tool(
                name="synthesize_data",
                description="直接调用 LLM 生成合成数据 (需要 API key)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "analysis_dir": {
                            "type": "string",
                            "description": "DataRecipe 分析输出目录的路径",
                        },
                        "output_path": {
                            "type": "string",
                            "description": "输出文件路径（可选）",
                        },
                        "count": {
                            "type": "integer",
                            "description": "生成数量 (默认: 100)",
                            "default": 100,
                        },
                        "model": {
                            "type": "string",
                            "description": "LLM 模型 (默认: claude-sonnet-4-20250514)",
                        },
                        "provider": {
                            "type": "string",
                            "enum": ["anthropic", "openai"],
                            "description": "LLM 提供商",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["json", "jsonl"],
                            "description": "输出格式 (默认: json)",
                        },
                        "data_type": {
                            "type": "string",
                            "enum": ["auto", "instruction_response", "preference", "multi_turn"],
                            "description": "数据类型 (默认: auto)",
                        },
                        "resume": {
                            "type": "boolean",
                            "description": "增量续跑 (默认: false)",
                            "default": False,
                        },
                    },
                    "required": ["analysis_dir"],
                },
            ),
            Tool(
                name="validate_data",
                description="验证数据文件是否符合 Schema",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data_path": {
                            "type": "string",
                            "description": "数据文件路径 (JSON / JSONL)",
                        },
                        "schema_path": {
                            "type": "string",
                            "description": "Schema 文件路径",
                        },
                    },
                    "required": ["data_path", "schema_path"],
                },
            ),
            Tool(
                name="synth_augment",
                description="对已有数据做变体扩增（改写/回译/扰动/风格迁移）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data_path": {
                            "type": "string",
                            "description": "源数据文件路径 (JSON / JSONL)",
                        },
                        "output_path": {
                            "type": "string",
                            "description": "输出文件路径",
                        },
                        "strategy": {
                            "type": "string",
                            "enum": ["paraphrase", "backtranslate", "perturb", "style_transfer", "all"],
                            "description": "扩增策略 (默认: paraphrase)",
                            "default": "paraphrase",
                        },
                        "multiplier": {
                            "type": "integer",
                            "description": "每条数据生成的变体数量 (默认: 3)",
                            "default": 3,
                        },
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "要扩增的文本字段名列表（默认自动检测）",
                        },
                        "preserve_labels": {
                            "type": "boolean",
                            "description": "保持标签/分类不变 (默认: true)",
                            "default": True,
                        },
                    },
                    "required": ["data_path", "output_path"],
                },
            ),
            Tool(
                name="synth_batch",
                description="批量合成数据（支持进度追踪和断点续传）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "analysis_dir": {
                            "type": "string",
                            "description": "DataRecipe 分析输出目录",
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "批量输出目录",
                        },
                        "total_count": {
                            "type": "integer",
                            "description": "总目标数量",
                        },
                        "batch_size": {
                            "type": "integer",
                            "description": "每批数量 (默认: 50)",
                            "default": 50,
                        },
                        "model": {
                            "type": "string",
                            "description": "LLM 模型 (默认: claude-sonnet-4-20250514)",
                        },
                        "provider": {
                            "type": "string",
                            "enum": ["anthropic", "openai"],
                            "description": "LLM 提供商",
                        },
                        "resume": {
                            "type": "boolean",
                            "description": "从已有进度续传 (默认: true)",
                            "default": True,
                        },
                    },
                    "required": ["analysis_dir", "output_dir", "total_count"],
                },
            ),
            Tool(
                name="synth_evaluate",
                description="对合成数据做多维度快检（多样性/忠实度/质量分布）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data_path": {
                            "type": "string",
                            "description": "合成数据文件路径 (JSON / JSONL)",
                        },
                        "schema_path": {
                            "type": "string",
                            "description": "Schema 文件路径（可选，有则验证合规性）",
                        },
                        "seed_path": {
                            "type": "string",
                            "description": "种子数据路径（可选，有则计算与种子的相似度）",
                        },
                        "metrics": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["diversity", "faithfulness", "quality", "dedup", "length_distribution", "all"],
                            },
                            "description": "评估维度 (默认: all)",
                        },
                        "sample_size": {
                            "type": "integer",
                            "description": "抽样评估数量（0=全量，默认: 200）",
                            "default": 200,
                        },
                    },
                    "required": ["data_path"],
                },
            ),
            Tool(
                name="estimate_synthesis_cost",
                description="估算合成成本",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "count": {
                            "type": "integer",
                            "description": "目标生成数量",
                        },
                        "model": {
                            "type": "string",
                            "description": "LLM 模型",
                        },
                    },
                    "required": ["count"],
                },
            ),
            Tool(
                name="synth_translate",
                description="将合成数据翻译为目标语言（保留格式和标签结构）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data_path": {
                            "type": "string",
                            "description": "输入数据文件路径 (JSON/JSONL)",
                        },
                        "target_lang": {
                            "type": "string",
                            "description": "目标语言 (zh/en/ja/ko/fr/de 等)",
                        },
                        "output_path": {
                            "type": "string",
                            "description": "输出文件路径",
                        },
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "要翻译的字段（可选，默认翻译所有文本字段）",
                        },
                    },
                    "required": ["data_path", "target_lang", "output_path"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """调用工具."""

        if name == "prepare_synthesis":
            analysis_dir = Path(arguments["analysis_dir"])
            count = arguments.get("count", 10)

            # Load schema
            schema_path = analysis_dir / "04_复刻指南" / "DATA_SCHEMA.json"
            if not schema_path.exists():
                return [TextContent(type="text", text=f"Schema 未找到: {schema_path}")]

            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)

            # Load seeds
            samples_path = analysis_dir / "09_样例数据" / "samples.json"
            if not samples_path.exists():
                return [TextContent(type="text", text=f"种子数据未找到: {samples_path}")]

            with open(samples_path, "r", encoding="utf-8") as f:
                samples_data = json.load(f)

            seed_samples = []
            for s in samples_data.get("samples", []):
                if "data" in s:
                    seed_samples.append(s["data"])
                else:
                    seed_samples.append(s)

            # Load guidelines
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
                data_type=arguments.get("data_type", "auto"),
            )

            return [
                TextContent(
                    type="text",
                    text=f"## 合成 Prompt\n\n"
                    f"请使用以下 Prompt 生成 {count} 条数据，然后使用 parse_synthesis_result 解析结果。\n\n"
                    f"---\n\n{result['prompt']}\n\n---\n\n"
                    f"Schema 路径: {schema_path}\n"
                    f"建议输出路径: {analysis_dir}/11_合成数据/synthetic.json",
                )
            ]

        elif name == "parse_synthesis_result":
            response_text = arguments["response_text"]
            schema_path = arguments["schema_path"]
            output_path = arguments["output_path"]

            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)

            synthesizer = InteractiveSynthesizer()
            result = synthesizer.parse_result(
                response_text=response_text,
                schema=schema,
                output_path=output_path,
            )

            if result.success:
                return [
                    TextContent(
                        type="text",
                        text=f"✓ 合成数据已保存:\n"
                        f"- 输出路径: {result.output_path}\n"
                        f"- 生成数量: {result.generated_count}",
                    )
                ]
            else:
                return [TextContent(type="text", text=f"解析失败: {result.error}")]

        elif name == "synthesize_data":
            config = SynthesisConfig(
                target_count=arguments.get("count", 100),
                model=arguments.get("model", "claude-sonnet-4-20250514"),
                provider=arguments.get("provider", "anthropic"),
                data_type=arguments.get("data_type", "auto"),
            )

            synthesizer = DataSynthesizer(config)
            result = synthesizer.synthesize_from_datarecipe(
                analysis_dir=arguments["analysis_dir"],
                output_path=arguments.get("output_path"),
                target_count=arguments.get("count"),
                output_format=arguments.get("format", "json"),
                resume=arguments.get("resume", False),
            )

            if result.success:
                lines = [
                    "✓ 合成完成:",
                    f"- 输出路径: {result.output_path}",
                    f"- 生成数量: {result.generated_count}",
                    f"- 失败数量: {result.failed_count}",
                ]
                if result.dedup_count:
                    lines.append(f"- 去重数量: {result.dedup_count}")
                lines.extend([
                    f"- Token 用量: {result.total_tokens:,}",
                    f"- 预计成本: ${result.estimated_cost:.4f}",
                    f"- 耗时: {result.duration_seconds:.1f}s",
                ])
                if result.stats:
                    lines.append(f"\n统计概要:")
                    lines.append(f"- 总样本数: {result.stats['total_samples']}")
                    for fname, fstat in result.stats.get("fields", {}).items():
                        ftype = fstat.get("type", "unknown")
                        if ftype == "text":
                            lines.append(
                                f"- {fname}: 平均长度 {fstat['avg_length']}, "
                                f"范围 [{fstat['min_length']}, {fstat['max_length']}]"
                            )
                        elif ftype == "numeric":
                            lines.append(
                                f"- {fname}: 平均 {fstat['avg']}, "
                                f"范围 [{fstat['min']}, {fstat['max']}]"
                            )
                return [TextContent(type="text", text="\n".join(lines))]
            else:
                return [TextContent(type="text", text=f"合成失败: {result.error}")]

        elif name == "validate_data":
            from datasynth.config import DataSchema

            data_path = Path(arguments["data_path"])
            schema_path = Path(arguments["schema_path"])

            if not schema_path.exists():
                return [TextContent(type="text", text=f"Schema 未找到: {schema_path}")]
            if not data_path.exists():
                return [TextContent(type="text", text=f"数据文件未找到: {data_path}")]

            with open(schema_path, "r", encoding="utf-8") as f:
                schema = DataSchema.from_dict(json.load(f))

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
                return [TextContent(type="text", text="数据文件为空")]

            valid = 0
            errors: list[tuple[int, list[str]]] = []
            for i, sample in enumerate(samples):
                errs = schema.validate_sample(sample)
                if errs:
                    errors.append((i + 1, errs))
                else:
                    valid += 1

            lines = [
                f"验证结果: {len(samples)} 条数据",
                f"- ✓ 合规: {valid}",
                f"- ✗ 不合规: {len(errors)}",
            ]
            if errors:
                lines.append("\n错误详情 (前 5 条):")
                for idx, errs in errors[:5]:
                    lines.append(f"  #{idx}: {'; '.join(errs)}")
                if len(errors) > 5:
                    lines.append(f"  ... 共 {len(errors)} 条错误")

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "synth_augment":
            data_path = Path(arguments["data_path"])
            output_path = Path(arguments["output_path"])
            strategy = arguments.get("strategy", "paraphrase")
            multiplier = arguments.get("multiplier", 3)
            fields = arguments.get("fields")
            preserve_labels = arguments.get("preserve_labels", True)

            if not data_path.exists():
                return [TextContent(type="text", text=f"数据文件未找到: {data_path}")]

            # Load source data
            samples: list = []
            with open(data_path, "r", encoding="utf-8") as f:
                if data_path.suffix == ".jsonl":
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line))
                else:
                    data = json.load(f)
                    samples = data if isinstance(data, list) else data.get("samples", [])

            if not samples:
                return [TextContent(type="text", text="源数据为空")]

            # Auto-detect text fields if not specified
            if not fields:
                fields = []
                sample = samples[0]
                for k, v in (sample.get("data", sample) if isinstance(sample, dict) else {}).items():
                    if isinstance(v, str) and len(v) > 20:
                        fields.append(k)
                if not fields:
                    fields = [k for k, v in (samples[0] if isinstance(samples[0], dict) else {}).items()
                             if isinstance(v, str) and len(v) > 20]

            # Build augmentation prompt
            strategy_desc = {
                "paraphrase": "改写：保持语义不变，换一种表达方式",
                "backtranslate": "回译：模拟翻译为其他语言后再翻回中文的效果",
                "perturb": "扰动：在细节上做小幅修改（数字、名称、顺序等）",
                "style_transfer": "风格迁移：改变语气（正式↔口语、简洁↔详细等）",
                "all": "综合使用以上所有策略",
            }
            prompt_lines = [
                "## 数据扩增 Prompt",
                "",
                f"策略: {strategy_desc.get(strategy, strategy)}",
                f"目标: 每条数据生成 {multiplier} 个变体",
                f"扩增字段: {', '.join(fields)}",
                f"保持标签不变: {'是' if preserve_labels else '否'}",
                f"源数据量: {len(samples)} 条",
                f"预计输出: {len(samples) * multiplier} 条变体",
                "",
                "---",
                "",
                f"请对以下 {min(3, len(samples))} 条样例数据的 `{', '.join(fields)}` 字段",
                f"各生成 {multiplier} 个变体。使用 JSON 数组输出，每个变体保留原始所有字段。",
                "",
                "样例数据:",
                "```json",
                json.dumps(samples[:3], ensure_ascii=False, indent=2),
                "```",
                "",
                f"输出路径: {output_path}",
            ]

            return [TextContent(type="text", text="\n".join(prompt_lines))]

        elif name == "synth_batch":
            analysis_dir = Path(arguments["analysis_dir"])
            output_dir = Path(arguments["output_dir"])
            total_count = arguments["total_count"]
            batch_size = arguments.get("batch_size", 50)
            resume = arguments.get("resume", True)

            output_dir.mkdir(parents=True, exist_ok=True)

            # Check existing progress
            existing = 0
            existing_files: list[str] = []
            if resume:
                for f in sorted(output_dir.glob("batch_*.jsonl")):
                    with open(f, "r", encoding="utf-8") as fh:
                        count = sum(1 for line in fh if line.strip())
                    existing += count
                    existing_files.append(f"{f.name}: {count} 条")

            remaining = max(0, total_count - existing)
            batches_needed = (remaining + batch_size - 1) // batch_size if remaining > 0 else 0

            config = SynthesisConfig(
                target_count=batch_size,
                model=arguments.get("model", "claude-sonnet-4-20250514"),
                provider=arguments.get("provider", "anthropic"),
            )
            cost_per_batch = config.estimate_cost()

            lines = [
                "## 批量合成计划",
                "",
                f"- 总目标: {total_count} 条",
                f"- 已有进度: {existing} 条" + (f"（{len(existing_files)} 个文件）" if existing_files else ""),
                f"- 剩余: {remaining} 条",
                f"- 批大小: {batch_size}",
                f"- 待执行批次: {batches_needed}",
                f"- 预计成本: ${cost_per_batch['estimated_cost_usd'] * batches_needed:.2f}",
                f"- 输出目录: {output_dir}",
            ]

            if remaining > 0:
                lines.extend([
                    "",
                    "执行方式: 逐批调用 `synthesize_data` 工具，每批指定:",
                    f"  analysis_dir={analysis_dir}",
                    f"  output_path={output_dir}/batch_{{N:03d}}.jsonl",
                    f"  count={batch_size}",
                    "  format=jsonl",
                    f"  resume={resume}",
                ])

                if existing_files:
                    lines.extend(["", "已有文件:"])
                    for ef in existing_files[:10]:
                        lines.append(f"  - {ef}")
                    if len(existing_files) > 10:
                        lines.append(f"  ... 共 {len(existing_files)} 个文件")
            else:
                lines.append("\n已达目标，无需继续合成。")

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "synth_evaluate":
            data_path = Path(arguments["data_path"])
            if not data_path.exists():
                return [TextContent(type="text", text=f"数据文件未找到: {data_path}")]

            # Load data
            samples: list = []
            with open(data_path, "r", encoding="utf-8") as f:
                if data_path.suffix == ".jsonl":
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line))
                else:
                    data = json.load(f)
                    samples = data if isinstance(data, list) else data.get("samples", [])

            if not samples:
                return [TextContent(type="text", text="数据文件为空")]

            metrics = arguments.get("metrics", ["all"])
            if "all" in metrics:
                metrics = ["diversity", "faithfulness", "quality", "dedup", "length_distribution"]

            sample_size = arguments.get("sample_size", 200)
            import random
            eval_samples = random.sample(samples, min(sample_size, len(samples))) if sample_size and sample_size < len(samples) else samples

            lines = [f"## 合成数据评估报告", "", f"文件: {data_path}", f"总量: {len(samples)} 条", f"评估样本: {len(eval_samples)} 条", ""]

            # Detect text fields
            text_fields: list[str] = []
            if eval_samples and isinstance(eval_samples[0], dict):
                for k, v in eval_samples[0].items():
                    if isinstance(v, str) and len(v) > 10:
                        text_fields.append(k)

            if "length_distribution" in metrics and text_fields:
                lines.append("### 长度分布")
                for field in text_fields:
                    lengths = [len(s.get(field, "")) for s in eval_samples if isinstance(s.get(field), str)]
                    if lengths:
                        avg_len = sum(lengths) / len(lengths)
                        min_len = min(lengths)
                        max_len = max(lengths)
                        lines.append(f"- **{field}**: 平均 {avg_len:.0f} 字, 范围 [{min_len}, {max_len}]")
                lines.append("")

            if "dedup" in metrics:
                lines.append("### 去重分析")
                # Simple dedup by first text field
                if text_fields:
                    field = text_fields[0]
                    texts = [s.get(field, "") for s in eval_samples if isinstance(s.get(field), str)]
                    unique = len(set(texts))
                    dup_rate = 1 - unique / len(texts) if texts else 0
                    lines.append(f"- 基于 `{field}` 字段: {unique}/{len(texts)} 唯一 (重复率 {dup_rate:.1%})")
                # Hash-based exact dedup
                hashes = set()
                exact_dups = 0
                for s in eval_samples:
                    h = hash(json.dumps(s, sort_keys=True, ensure_ascii=False))
                    if h in hashes:
                        exact_dups += 1
                    hashes.add(h)
                lines.append(f"- 完全相同记录: {exact_dups}/{len(eval_samples)}")
                lines.append("")

            if "diversity" in metrics and text_fields:
                lines.append("### 多样性")
                field = text_fields[0]
                texts = [s.get(field, "") for s in eval_samples if isinstance(s.get(field), str)]
                if texts:
                    # Vocabulary diversity
                    all_chars = set()
                    all_words: set[str] = set()
                    for t in texts:
                        all_chars.update(t)
                        all_words.update(t.split())
                    lines.append(f"- 基于 `{field}` 字段:")
                    lines.append(f"  - 唯一字符数: {len(all_chars)}")
                    lines.append(f"  - 唯一词汇数: {len(all_words)}")
                    # Starting pattern diversity
                    starts = set(t[:20] for t in texts if len(t) >= 20)
                    lines.append(f"  - 开头模式多样性: {len(starts)}/{len(texts)} ({len(starts)/len(texts):.1%})")
                lines.append("")

            if "quality" in metrics:
                lines.append("### 质量检查")
                # Basic quality metrics
                empty_count = sum(1 for s in eval_samples if not s or (isinstance(s, dict) and not any(s.values())))
                lines.append(f"- 空记录: {empty_count}/{len(eval_samples)}")
                if text_fields:
                    for field in text_fields[:3]:
                        short = sum(1 for s in eval_samples if isinstance(s.get(field), str) and len(s[field]) < 10)
                        lines.append(f"- `{field}` 过短 (<10字): {short}/{len(eval_samples)}")
                lines.append("")

            if "faithfulness" in metrics:
                schema_path = arguments.get("schema_path")
                seed_path = arguments.get("seed_path")
                lines.append("### 忠实度")
                if schema_path and Path(schema_path).exists():
                    from datasynth.config import DataSchema
                    with open(schema_path, "r", encoding="utf-8") as f:
                        schema = DataSchema.from_dict(json.load(f))
                    errors = 0
                    for s in eval_samples:
                        if schema.validate_sample(s):
                            errors += 1
                    lines.append(f"- Schema 合规率: {(len(eval_samples) - errors)}/{len(eval_samples)} ({(1 - errors/len(eval_samples)):.1%})")
                else:
                    lines.append("- Schema 合规率: 未提供 schema_path，跳过")
                if seed_path and Path(seed_path).exists():
                    lines.append("- 种子相似度: 需要 embedding 计算，建议使用专用工具")
                else:
                    lines.append("- 种子相似度: 未提供 seed_path，跳过")
                lines.append("")

            lines.append("---")
            lines.append(f"评估维度: {', '.join(metrics)}")

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "estimate_synthesis_cost":
            config = SynthesisConfig(
                target_count=arguments["count"],
                model=arguments.get("model", "claude-sonnet-4-20250514"),
            )
            estimate = config.estimate_cost()

            return [
                TextContent(
                    type="text",
                    text=f"成本估算:\n"
                    f"- 目标数量: {estimate['target_count']}\n"
                    f"- 预计批次: {estimate['estimated_batches']}\n"
                    f"- 预计 Token: {estimate['estimated_input_tokens'] + estimate['estimated_output_tokens']:,}\n"
                    f"- 预计成本: ${estimate['estimated_cost_usd']:.2f}\n"
                    f"- 模型: {estimate['model']}",
                )
            ]

        elif name == "synth_translate":
            data_path = Path(arguments["data_path"])
            target_lang = arguments["target_lang"]
            output_path = arguments["output_path"]
            fields = arguments.get("fields")

            if not data_path.exists():
                return [TextContent(type="text", text=f"文件不存在: {data_path}")]

            # Load samples
            samples = []
            suffix = data_path.suffix.lower()
            with open(data_path, "r", encoding="utf-8") as f:
                if suffix == ".jsonl":
                    for line in f:
                        line = line.strip()
                        if line:
                            samples.append(json.loads(line))
                else:
                    samples = json.load(f)
                    if isinstance(samples, dict):
                        samples = samples.get("data", samples.get("samples", [samples]))

            if not samples:
                return [TextContent(type="text", text="错误: 数据文件为空")]

            # Auto-detect text fields if not specified
            if not fields:
                sample = samples[0]
                fields = [k for k, v in sample.items() if isinstance(v, str) and len(v) > 5]

            lang_names = {
                "zh": "中文", "en": "English", "ja": "日本語", "ko": "한국어",
                "fr": "Français", "de": "Deutsch", "es": "Español",
            }
            lang_display = lang_names.get(target_lang, target_lang)

            # Generate translation prompt instead of doing actual translation
            lines = [
                f"## 翻译任务生成",
                "",
                f"- 源文件: `{data_path.name}` ({len(samples)} 条)",
                f"- 目标语言: {lang_display} ({target_lang})",
                f"- 翻译字段: {', '.join(fields)}",
                f"- 输出路径: {output_path}",
                "",
                "### 翻译提示模板",
                "",
                "```",
                f"请将以下 JSON 数据中的 {', '.join(fields)} 字段翻译为{lang_display}，",
                "保持 JSON 格式和其他字段不变。",
                "",
                "输入样例:",
                json.dumps(samples[0], ensure_ascii=False, indent=2)[:500],
                "```",
                "",
                f"共 {len(samples)} 条待翻译。建议使用 LLM 批量处理。",
            ]

            return [TextContent(type="text", text="\n".join(lines))]

        else:
            return [TextContent(type="text", text=f"未知工具: {name}")]

    return server


async def serve():
    """启动 MCP 服务器."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install datasynth[mcp]")

    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


def main():
    """主入口."""
    import asyncio

    asyncio.run(serve())


if __name__ == "__main__":
    main()
