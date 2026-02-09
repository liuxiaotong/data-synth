"""Tests for MCP server tools."""

import json
from unittest.mock import patch

import pytest

try:
    from mcp.types import CallToolRequest, CallToolRequestParams

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

pytestmark = pytest.mark.skipif(not HAS_MCP, reason="MCP not installed")


@pytest.fixture
def datarecipe_dir(tmp_path):
    schema_dir = tmp_path / "04_复刻指南"
    schema_dir.mkdir()
    (schema_dir / "DATA_SCHEMA.json").write_text(
        json.dumps(
            {
                "project_name": "MCP测试",
                "fields": [
                    {"name": "instruction", "type": "text"},
                    {"name": "response", "type": "text"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    samples_dir = tmp_path / "09_样例数据"
    samples_dir.mkdir()
    (samples_dir / "samples.json").write_text(
        json.dumps(
            {
                "samples": [
                    {"data": {"instruction": "Q1", "response": "A1"}},
                    {"data": {"instruction": "Q2", "response": "A2"}},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return tmp_path


async def _call_tool(server, name, arguments):
    """Call a tool on the MCP server and return content list."""
    handler = server.request_handlers[CallToolRequest]
    req = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name=name, arguments=arguments),
    )
    result = await handler(req)
    return result.root.content


@pytest.fixture
def server():
    from datasynth.mcp_server import create_server

    return create_server()


class TestListTools:
    @pytest.mark.asyncio
    async def test_list_tools(self, server):
        from mcp.types import ListToolsRequest

        handler = server.request_handlers[ListToolsRequest]
        req = ListToolsRequest(method="tools/list")
        result = await handler(req)
        tools = result.root.tools
        names = {t.name for t in tools}
        assert names == {
            "prepare_synthesis",
            "parse_synthesis_result",
            "synthesize_data",
            "estimate_synthesis_cost",
        }


class TestEstimateSynthesisCost:
    @pytest.mark.asyncio
    async def test_estimate(self, server):
        content = await _call_tool(server, "estimate_synthesis_cost", {"count": 100})
        text = content[0].text
        assert "成本估算" in text
        assert "100" in text

    @pytest.mark.asyncio
    async def test_estimate_custom_model(self, server):
        content = await _call_tool(
            server, "estimate_synthesis_cost", {"count": 50, "model": "gpt-4o"}
        )
        assert "gpt-4o" in content[0].text


class TestPrepareSynthesis:
    @pytest.mark.asyncio
    async def test_prepare(self, server, datarecipe_dir):
        content = await _call_tool(
            server,
            "prepare_synthesis",
            {"analysis_dir": str(datarecipe_dir), "count": 5},
        )
        text = content[0].text
        assert "合成 Prompt" in text
        assert "MCP测试" in text

    @pytest.mark.asyncio
    async def test_prepare_missing_schema(self, server, tmp_path):
        content = await _call_tool(
            server, "prepare_synthesis", {"analysis_dir": str(tmp_path)}
        )
        assert "Schema 未找到" in content[0].text

    @pytest.mark.asyncio
    async def test_prepare_missing_samples(self, server, tmp_path):
        schema_dir = tmp_path / "04_复刻指南"
        schema_dir.mkdir()
        (schema_dir / "DATA_SCHEMA.json").write_text("{}", encoding="utf-8")

        content = await _call_tool(
            server, "prepare_synthesis", {"analysis_dir": str(tmp_path)}
        )
        assert "种子数据未找到" in content[0].text


class TestParseSynthesisResult:
    @pytest.mark.asyncio
    async def test_parse(self, server, datarecipe_dir, tmp_path):
        schema_path = datarecipe_dir / "04_复刻指南" / "DATA_SCHEMA.json"
        output_path = tmp_path / "parsed.json"

        response_text = json.dumps(
            [{"instruction": "Q", "response": "A"}], ensure_ascii=False
        )

        content = await _call_tool(
            server,
            "parse_synthesis_result",
            {
                "response_text": response_text,
                "schema_path": str(schema_path),
                "output_path": str(output_path),
            },
        )

        assert "合成数据已保存" in content[0].text
        assert output_path.exists()

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(data["samples"]) == 1


class TestSynthesizeData:
    @pytest.mark.asyncio
    async def test_synthesize_success(self, server, datarecipe_dir):
        from datasynth.synthesizer import SynthesisResult

        mock_result = SynthesisResult(
            success=True,
            output_path=str(datarecipe_dir / "out.json"),
            generated_count=5,
            failed_count=0,
            total_tokens=1000,
            estimated_cost=0.01,
            duration_seconds=1.0,
        )

        with patch("datasynth.mcp_server.DataSynthesizer") as MockSynth:
            MockSynth.return_value.synthesize_from_datarecipe.return_value = mock_result
            content = await _call_tool(
                server, "synthesize_data", {"analysis_dir": str(datarecipe_dir), "count": 5}
            )

        assert "合成完成" in content[0].text

    @pytest.mark.asyncio
    async def test_synthesize_with_stats(self, server, datarecipe_dir):
        from datasynth.synthesizer import SynthesisResult

        mock_result = SynthesisResult(
            success=True,
            output_path=str(datarecipe_dir / "out.json"),
            generated_count=5,
            failed_count=0,
            dedup_count=1,
            total_tokens=1000,
            estimated_cost=0.01,
            duration_seconds=1.0,
            stats={
                "total_samples": 5,
                "fields": {
                    "instruction": {"type": "text", "avg_length": 20.0, "min_length": 5, "max_length": 50, "count": 5, "missing": 0},
                    "response": {"type": "text", "avg_length": 100.0, "min_length": 30, "max_length": 200, "count": 5, "missing": 0},
                },
            },
        )

        with patch("datasynth.mcp_server.DataSynthesizer") as MockSynth:
            MockSynth.return_value.synthesize_from_datarecipe.return_value = mock_result
            content = await _call_tool(
                server, "synthesize_data", {"analysis_dir": str(datarecipe_dir), "count": 5}
            )

        text = content[0].text
        assert "合成完成" in text
        assert "去重数量: 1" in text
        assert "统计概要" in text
        assert "instruction" in text
        assert "平均长度" in text

    @pytest.mark.asyncio
    async def test_synthesize_failure(self, server, datarecipe_dir):
        from datasynth.synthesizer import SynthesisResult

        mock_result = SynthesisResult(success=False, error="No API key")

        with patch("datasynth.mcp_server.DataSynthesizer") as MockSynth:
            MockSynth.return_value.synthesize_from_datarecipe.return_value = mock_result
            content = await _call_tool(
                server, "synthesize_data", {"analysis_dir": str(datarecipe_dir)}
            )

        assert "合成失败" in content[0].text


class TestUnknownTool:
    @pytest.mark.asyncio
    async def test_unknown(self, server):
        content = await _call_tool(server, "nonexistent_tool", {})
        assert "未知工具" in content[0].text
