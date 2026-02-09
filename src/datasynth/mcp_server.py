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

            synthesizer = InteractiveSynthesizer()
            result = synthesizer.prepare_synthesis(
                schema=schema,
                seed_samples=seed_samples,
                count=count,
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
                return [
                    TextContent(
                        type="text",
                        text=f"✓ 合成完成:\n"
                        f"- 输出路径: {result.output_path}\n"
                        f"- 生成数量: {result.generated_count}\n"
                        f"- 失败数量: {result.failed_count}\n"
                        f"- Token 用量: {result.total_tokens:,}\n"
                        f"- 预计成本: ${result.estimated_cost:.4f}\n"
                        f"- 耗时: {result.duration_seconds:.1f}s",
                    )
                ]
            else:
                return [TextContent(type="text", text=f"合成失败: {result.error}")]

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
