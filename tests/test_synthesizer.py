"""Tests for DataSynthesizer and InteractiveSynthesizer."""

import json
from unittest.mock import MagicMock, patch

import pytest

from datasynth.synthesizer import DataSynthesizer, InteractiveSynthesizer, SynthesisResult
from datasynth.config import SynthesisConfig


SAMPLE_SCHEMA = {
    "project_name": "测试项目",
    "fields": [
        {"name": "instruction", "type": "text"},
        {"name": "response", "type": "text"},
    ],
}

SAMPLE_SEEDS = [
    {"instruction": "什么是 AI？", "response": "AI 是人工智能的缩写..."},
    {"instruction": "解释机器学习", "response": "机器学习是..."},
    {"instruction": "什么是深度学习", "response": "深度学习是..."},
]

LLM_RESPONSE = json.dumps(
    [
        {"instruction": "什么是 NLP？", "response": "NLP 是自然语言处理..."},
        {"instruction": "解释强化学习", "response": "强化学习是..."},
    ],
    ensure_ascii=False,
)


class TestSynthesisResult:
    def test_defaults(self):
        r = SynthesisResult()
        assert r.success is True
        assert r.generated_count == 0
        assert r.error == ""


class TestDataSynthesizer:
    def test_init_default_config(self):
        s = DataSynthesizer()
        assert s.config.target_count == 100

    def test_init_custom_config(self):
        cfg = SynthesisConfig(target_count=50, batch_size=10)
        s = DataSynthesizer(cfg)
        assert s.config.target_count == 50
        assert s.config.batch_size == 10

    def test_init_client_anthropic(self):
        s = DataSynthesizer(SynthesisConfig(provider="anthropic"))
        with patch("datasynth.synthesizer.anthropic", create=True) as mock_mod:
            mock_mod.Anthropic.return_value = MagicMock()
            # Simulate the import inside _init_client
            with patch.dict("sys.modules", {"anthropic": mock_mod}):
                s._init_client()
                assert s._provider == "anthropic"

    def test_init_client_unsupported(self):
        s = DataSynthesizer(SynthesisConfig(provider="unsupported"))
        with pytest.raises(ValueError, match="不支持的 provider"):
            s._init_client()

    def test_synthesize_success(self, tmp_path):
        """Test full synthesize flow with mocked LLM."""
        cfg = SynthesisConfig(target_count=2, batch_size=2, provider="anthropic")
        s = DataSynthesizer(cfg)

        # Mock _init_client and _call_llm
        s._client = MagicMock()
        s._provider = "anthropic"

        s._call_llm = MagicMock(return_value=(LLM_RESPONSE, 500))

        output_path = tmp_path / "output.json"
        result = s.synthesize(
            schema=SAMPLE_SCHEMA,
            seed_samples=SAMPLE_SEEDS,
            output_path=str(output_path),
            target_count=2,
        )

        assert result.success
        assert result.generated_count == 2
        assert result.total_tokens == 500
        assert output_path.exists()

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data["metadata"]["generator"] == "DataSynth"
        assert len(data["samples"]) == 2
        assert data["samples"][0]["id"] == "SYNTH_0001"
        assert data["samples"][0]["synthetic"] is True

    def test_synthesize_jsonl_output(self, tmp_path):
        """Test JSONL output format."""
        cfg = SynthesisConfig(target_count=2, batch_size=2)
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        s._call_llm = MagicMock(return_value=(LLM_RESPONSE, 500))

        output_path = tmp_path / "output.jsonl"
        result = s.synthesize(
            schema=SAMPLE_SCHEMA,
            seed_samples=SAMPLE_SEEDS,
            output_path=str(output_path),
            target_count=2,
            output_format="jsonl",
        )

        assert result.success
        assert result.generated_count == 2

        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            obj = json.loads(line)
            assert "instruction" in obj
            assert "response" in obj

    def test_synthesize_with_progress(self, tmp_path):
        """Test progress callback is invoked."""
        cfg = SynthesisConfig(target_count=2, batch_size=2)
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        s._call_llm = MagicMock(return_value=(LLM_RESPONSE, 100))

        progress_calls = []
        result = s.synthesize(
            schema=SAMPLE_SCHEMA,
            seed_samples=SAMPLE_SEEDS,
            output_path=str(tmp_path / "out.json"),
            target_count=2,
            on_progress=lambda cur, tot: progress_calls.append((cur, tot)),
        )

        assert result.success
        assert len(progress_calls) >= 1

    def test_synthesize_llm_failure_exhausts_retries(self, tmp_path):
        """Test that LLM errors retry max_retries times then fail the batch."""
        cfg = SynthesisConfig(target_count=2, batch_size=2, max_retries=3, retry_delay=0)
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        s._call_llm = MagicMock(side_effect=RuntimeError("API error"))

        result = s.synthesize(
            schema=SAMPLE_SCHEMA,
            seed_samples=SAMPLE_SEEDS,
            output_path=str(tmp_path / "out.json"),
            target_count=2,
        )

        assert s._call_llm.call_count == 3  # retried 3 times

        assert result.success  # overall still succeeds, just with 0 generated
        assert result.generated_count == 0
        assert result.failed_count == 2

    def test_synthesize_retry_succeeds(self, tmp_path):
        """Test that a batch succeeds after transient failure."""
        cfg = SynthesisConfig(target_count=2, batch_size=2, max_retries=3, retry_delay=0)
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        # Fail first, succeed second
        s._call_llm = MagicMock(
            side_effect=[RuntimeError("timeout"), (LLM_RESPONSE, 500)]
        )

        result = s.synthesize(
            schema=SAMPLE_SCHEMA,
            seed_samples=SAMPLE_SEEDS,
            output_path=str(tmp_path / "out.json"),
            target_count=2,
        )

        assert result.success
        assert result.generated_count == 2
        assert result.failed_count == 0
        assert s._call_llm.call_count == 2

    def test_synthesize_from_datarecipe(self, tmp_path):
        """Test synthesize_from_datarecipe reads correct files."""
        # Set up directory structure
        schema_dir = tmp_path / "04_复刻指南"
        schema_dir.mkdir()
        (schema_dir / "DATA_SCHEMA.json").write_text(
            json.dumps(SAMPLE_SCHEMA, ensure_ascii=False), encoding="utf-8"
        )

        samples_dir = tmp_path / "09_样例数据"
        samples_dir.mkdir()
        (samples_dir / "samples.json").write_text(
            json.dumps({"samples": [{"data": s} for s in SAMPLE_SEEDS]}, ensure_ascii=False),
            encoding="utf-8",
        )

        cfg = SynthesisConfig(target_count=2, batch_size=2)
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        s._call_llm = MagicMock(return_value=(LLM_RESPONSE, 200))

        result = s.synthesize_from_datarecipe(
            analysis_dir=str(tmp_path),
            target_count=2,
        )

        assert result.success
        assert result.generated_count == 2
        output = tmp_path / "11_合成数据" / "synthetic.json"
        assert output.exists()

    def test_synthesize_from_datarecipe_jsonl(self, tmp_path):
        """Test synthesize_from_datarecipe with JSONL defaults to .jsonl extension."""
        schema_dir = tmp_path / "04_复刻指南"
        schema_dir.mkdir()
        (schema_dir / "DATA_SCHEMA.json").write_text(
            json.dumps(SAMPLE_SCHEMA, ensure_ascii=False), encoding="utf-8"
        )

        samples_dir = tmp_path / "09_样例数据"
        samples_dir.mkdir()
        (samples_dir / "samples.json").write_text(
            json.dumps({"samples": [{"data": s} for s in SAMPLE_SEEDS]}, ensure_ascii=False),
            encoding="utf-8",
        )

        cfg = SynthesisConfig(target_count=2, batch_size=2)
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        s._call_llm = MagicMock(return_value=(LLM_RESPONSE, 200))

        result = s.synthesize_from_datarecipe(
            analysis_dir=str(tmp_path),
            target_count=2,
            output_format="jsonl",
        )

        assert result.success
        output = tmp_path / "11_合成数据" / "synthetic.jsonl"
        assert output.exists()
        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_synthesize_from_datarecipe_missing_schema(self, tmp_path):
        result = DataSynthesizer().synthesize_from_datarecipe(str(tmp_path))
        assert not result.success
        assert "Schema not found" in result.error

    def test_synthesize_from_datarecipe_missing_samples(self, tmp_path):
        schema_dir = tmp_path / "04_复刻指南"
        schema_dir.mkdir()
        (schema_dir / "DATA_SCHEMA.json").write_text("{}", encoding="utf-8")

        result = DataSynthesizer().synthesize_from_datarecipe(str(tmp_path))
        assert not result.success
        assert "Seed samples not found" in result.error

    def test_synthesize_from_datarecipe_empty_samples(self, tmp_path):
        schema_dir = tmp_path / "04_复刻指南"
        schema_dir.mkdir()
        (schema_dir / "DATA_SCHEMA.json").write_text(
            json.dumps(SAMPLE_SCHEMA), encoding="utf-8"
        )

        samples_dir = tmp_path / "09_样例数据"
        samples_dir.mkdir()
        (samples_dir / "samples.json").write_text(
            json.dumps({"samples": []}), encoding="utf-8"
        )

        result = DataSynthesizer().synthesize_from_datarecipe(str(tmp_path))
        assert not result.success
        assert "No seed samples" in result.error

    def test_synthesize_from_datarecipe_with_guidelines(self, tmp_path):
        """Test that guidelines file is loaded when present."""
        schema_dir = tmp_path / "04_复刻指南"
        schema_dir.mkdir()
        (schema_dir / "DATA_SCHEMA.json").write_text(
            json.dumps(SAMPLE_SCHEMA, ensure_ascii=False), encoding="utf-8"
        )

        samples_dir = tmp_path / "09_样例数据"
        samples_dir.mkdir()
        (samples_dir / "samples.json").write_text(
            json.dumps({"samples": [{"data": s} for s in SAMPLE_SEEDS]}, ensure_ascii=False),
            encoding="utf-8",
        )

        guidelines_dir = tmp_path / "03_标注规范"
        guidelines_dir.mkdir()
        (guidelines_dir / "ANNOTATION_SPEC.md").write_text(
            "请确保数据质量", encoding="utf-8"
        )

        cfg = SynthesisConfig(target_count=2, batch_size=2)
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        s._call_llm = MagicMock(return_value=(LLM_RESPONSE, 200))

        result = s.synthesize_from_datarecipe(str(tmp_path), target_count=2)
        assert result.success

        # Verify guidelines were passed to the prompt
        call_args = s._call_llm.call_args[0][0]
        assert "数据质量" in call_args

    def test_estimate_cost(self):
        s = DataSynthesizer()
        cost = s._estimate_cost(10000)
        assert cost > 0


class TestInteractiveSynthesizer:
    def test_prepare_synthesis(self):
        s = InteractiveSynthesizer()
        result = s.prepare_synthesis(
            schema=SAMPLE_SCHEMA,
            seed_samples=SAMPLE_SEEDS,
            count=5,
        )

        assert "prompt" in result
        assert result["expected_count"] == 5
        assert "测试项目" in result["prompt"]
        assert "5" in result["prompt"]

    def test_prepare_synthesis_with_guidelines(self):
        s = InteractiveSynthesizer()
        result = s.prepare_synthesis(
            schema=SAMPLE_SCHEMA,
            seed_samples=SAMPLE_SEEDS,
            count=5,
            guidelines="保持简洁",
        )

        assert "简洁" in result["prompt"]

    def test_parse_result(self, tmp_path):
        s = InteractiveSynthesizer()
        response = """```json
[
  {"instruction": "问题1", "response": "回答1"},
  {"instruction": "问题2", "response": "回答2"}
]
```"""
        output_path = tmp_path / "sub" / "output.json"
        result = s.parse_result(
            response_text=response,
            schema=SAMPLE_SCHEMA,
            output_path=str(output_path),
        )

        assert result.success
        assert result.generated_count == 2
        assert output_path.exists()

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data["metadata"]["generator"] == "DataSynth (Interactive)"
        assert len(data["samples"]) == 2

    def test_parse_result_empty(self, tmp_path):
        s = InteractiveSynthesizer()
        result = s.parse_result(
            response_text="no json here",
            schema=SAMPLE_SCHEMA,
            output_path=str(tmp_path / "out.json"),
        )

        assert result.success
        assert result.generated_count == 0
