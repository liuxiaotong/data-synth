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

    def test_synthesize_concurrent(self, tmp_path):
        """Test concurrent batch generation."""
        cfg = SynthesisConfig(target_count=4, batch_size=2, concurrency=2)
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        s._call_llm = MagicMock(return_value=(LLM_RESPONSE, 250))

        result = s.synthesize(
            schema=SAMPLE_SCHEMA,
            seed_samples=SAMPLE_SEEDS,
            output_path=str(tmp_path / "out.json"),
            target_count=4,
        )

        assert result.success
        # 2 batches x 2 samples, but dedup removes duplicates across batches
        # (both batches return same LLM_RESPONSE)
        assert result.generated_count == 2  # deduped across batches
        assert result.dedup_count == 2
        assert s._call_llm.call_count == 2  # 2 batches

    def test_synthesize_concurrent_no_dedup(self, tmp_path):
        """Test concurrent generation without dedup."""
        cfg = SynthesisConfig(target_count=4, batch_size=2, concurrency=2, validate=False)
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        s._call_llm = MagicMock(return_value=(LLM_RESPONSE, 250))

        result = s.synthesize(
            schema=SAMPLE_SCHEMA,
            seed_samples=SAMPLE_SEEDS,
            output_path=str(tmp_path / "out.json"),
            target_count=4,
        )

        assert result.success
        assert result.generated_count == 4  # no dedup
        assert result.dedup_count == 0

    def test_dedup_within_batch(self, tmp_path):
        """Test that duplicate samples within a batch are removed."""
        duped_response = json.dumps(
            [
                {"instruction": "Q1", "response": "A1"},
                {"instruction": "Q1", "response": "A1"},  # duplicate
                {"instruction": "Q2", "response": "A2"},
            ],
            ensure_ascii=False,
        )
        cfg = SynthesisConfig(target_count=3, batch_size=3)
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        s._call_llm = MagicMock(return_value=(duped_response, 300))

        result = s.synthesize(
            schema=SAMPLE_SCHEMA,
            seed_samples=SAMPLE_SEEDS,
            output_path=str(tmp_path / "out.json"),
            target_count=3,
        )

        assert result.generated_count == 2
        assert result.dedup_count == 1

    def test_dedup_against_seeds(self, tmp_path):
        """Test that samples matching seed data are removed."""
        # Return a sample identical to a seed
        seed_copy_response = json.dumps(
            [
                {"instruction": "什么是 AI？", "response": "AI 是人工智能的缩写..."},  # same as seed
                {"instruction": "新问题", "response": "新回答"},
            ],
            ensure_ascii=False,
        )
        cfg = SynthesisConfig(target_count=2, batch_size=2)
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        s._call_llm = MagicMock(return_value=(seed_copy_response, 200))

        result = s.synthesize(
            schema=SAMPLE_SCHEMA,
            seed_samples=SAMPLE_SEEDS,
            output_path=str(tmp_path / "out.json"),
            target_count=2,
        )

        assert result.generated_count == 1
        assert result.dedup_count == 1

    def test_dedup_disabled(self, tmp_path):
        """Test that dedup is skipped when validate=False."""
        duped_response = json.dumps(
            [
                {"instruction": "Q1", "response": "A1"},
                {"instruction": "Q1", "response": "A1"},  # duplicate
            ],
            ensure_ascii=False,
        )
        cfg = SynthesisConfig(target_count=2, batch_size=2, validate=False)
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        s._call_llm = MagicMock(return_value=(duped_response, 200))

        result = s.synthesize(
            schema=SAMPLE_SCHEMA,
            seed_samples=SAMPLE_SEEDS,
            output_path=str(tmp_path / "out.json"),
            target_count=2,
        )

        assert result.generated_count == 2  # duplicates kept
        assert result.dedup_count == 0

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

    def test_schema_validation_filters_invalid(self, tmp_path):
        """Test that invalid samples are filtered out by schema validation."""
        schema_with_int = {
            "project_name": "测试项目",
            "fields": [
                {"name": "instruction", "type": "text"},
                {"name": "response", "type": "text"},
                {"name": "quality", "type": "int", "constraints": {"range": [1, 5]}},
            ],
        }
        # One valid, one with wrong type, one out of range
        response = json.dumps(
            [
                {"instruction": "Q1", "response": "A1", "quality": 4},
                {"instruction": "Q2", "response": "A2", "quality": "high"},  # invalid type
                {"instruction": "Q3", "response": "A3", "quality": 10},  # out of range
            ],
            ensure_ascii=False,
        )
        cfg = SynthesisConfig(target_count=3, batch_size=3, validate=True)
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        s._call_llm = MagicMock(return_value=(response, 300))

        result = s.synthesize(
            schema=schema_with_int,
            seed_samples=[{"instruction": "seed", "response": "ans", "quality": 3}],
            output_path=str(tmp_path / "out.json"),
            target_count=3,
        )

        assert result.success
        assert result.generated_count == 1  # only the valid one
        assert result.failed_count == 2  # two invalid

    def test_schema_validation_skipped_when_disabled(self, tmp_path):
        """Test that validation is skipped when validate=False."""
        schema_with_int = {
            "project_name": "测试项目",
            "fields": [
                {"name": "instruction", "type": "text"},
                {"name": "quality", "type": "int", "constraints": {"range": [1, 5]}},
            ],
        }
        response = json.dumps(
            [
                {"instruction": "Q1", "quality": 4},
                {"instruction": "Q2", "quality": "bad"},  # would be invalid
            ],
            ensure_ascii=False,
        )
        cfg = SynthesisConfig(target_count=2, batch_size=2, validate=False)
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        s._call_llm = MagicMock(return_value=(response, 200))

        result = s.synthesize(
            schema=schema_with_int,
            seed_samples=[{"instruction": "seed", "quality": 3}],
            output_path=str(tmp_path / "out.json"),
            target_count=2,
        )

        assert result.success
        assert result.generated_count == 2  # no validation, all pass

    def test_retry_temperature_increases(self, tmp_path):
        """Test that temperature increases on retry attempts."""
        cfg = SynthesisConfig(
            target_count=2, batch_size=2, max_retries=3, retry_delay=0, temperature=0.8
        )
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"

        # First call fails, second succeeds
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
        # First call: temperature=None (uses config default)
        # Second call: temperature=0.85 (0.8 + 0.05)
        first_call_temp = s._call_llm.call_args_list[0][1].get("temperature")
        second_call_temp = s._call_llm.call_args_list[1][1].get("temperature")
        assert first_call_temp is None  # first attempt uses default
        assert second_call_temp == pytest.approx(0.85, abs=0.001)

    def test_specialized_prompt_instruction_response(self, tmp_path):
        """Test that instruction_response schema uses specialized prompt."""
        cfg = SynthesisConfig(target_count=2, batch_size=2, data_type="auto")
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        s._call_llm = MagicMock(return_value=(LLM_RESPONSE, 500))

        result = s.synthesize(
            schema=SAMPLE_SCHEMA,
            seed_samples=SAMPLE_SEEDS,
            output_path=str(tmp_path / "out.json"),
            target_count=2,
        )

        assert result.success
        # Should use specialized prompt (contains "指令-回复数据生成")
        prompt_used = s._call_llm.call_args[0][0]
        assert "指令-回复" in prompt_used

    def test_specialized_prompt_includes_guidelines(self, tmp_path):
        """Specialized prompt should include guidelines when present."""
        cfg = SynthesisConfig(target_count=2, batch_size=2, data_type="auto")
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        s._call_llm = MagicMock(return_value=(LLM_RESPONSE, 500))

        result = s.synthesize(
            schema=SAMPLE_SCHEMA,
            seed_samples=SAMPLE_SEEDS,
            output_path=str(tmp_path / "out.json"),
            target_count=2,
            guidelines="自定义指南",
        )

        assert result.success
        prompt_used = s._call_llm.call_args[0][0]
        assert "自定义指南" in prompt_used
        assert "指令-回复" in prompt_used  # still uses specialized template

    def test_resume_continues_from_existing(self, tmp_path):
        """Test resume loads existing and generates remaining."""
        output_path = tmp_path / "out.json"
        # Write 2 existing samples
        existing = {
            "metadata": {},
            "schema": SAMPLE_SCHEMA,
            "samples": [
                {"id": "SYNTH_0001", "data": {"instruction": "已有Q1", "response": "已有A1"}, "synthetic": True},
                {"id": "SYNTH_0002", "data": {"instruction": "已有Q2", "response": "已有A2"}, "synthetic": True},
            ],
        }
        output_path.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf-8")

        cfg = SynthesisConfig(target_count=4, batch_size=2)
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        s._call_llm = MagicMock(return_value=(LLM_RESPONSE, 300))

        result = s.synthesize(
            schema=SAMPLE_SCHEMA,
            seed_samples=SAMPLE_SEEDS,
            output_path=str(output_path),
            target_count=4,
            resume=True,
        )

        assert result.success
        assert result.generated_count == 4  # 2 existing + 2 new

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(data["samples"]) == 4

    def test_resume_already_complete(self, tmp_path):
        """Test resume when target already met does nothing."""
        output_path = tmp_path / "out.json"
        existing = {
            "metadata": {},
            "schema": SAMPLE_SCHEMA,
            "samples": [
                {"id": "SYNTH_0001", "data": {"instruction": "Q1", "response": "A1"}, "synthetic": True},
                {"id": "SYNTH_0002", "data": {"instruction": "Q2", "response": "A2"}, "synthetic": True},
            ],
        }
        output_path.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf-8")

        cfg = SynthesisConfig(target_count=2, batch_size=2)
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        s._call_llm = MagicMock()

        result = s.synthesize(
            schema=SAMPLE_SCHEMA,
            seed_samples=SAMPLE_SEEDS,
            output_path=str(output_path),
            target_count=2,
            resume=True,
        )

        assert result.success
        assert result.generated_count == 2
        s._call_llm.assert_not_called()  # no LLM calls needed

    def test_resume_jsonl(self, tmp_path):
        """Test resume with JSONL format."""
        output_path = tmp_path / "out.jsonl"
        # Write 1 existing line
        output_path.write_text(
            json.dumps({"instruction": "已有Q", "response": "已有A"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        cfg = SynthesisConfig(target_count=3, batch_size=2)
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        s._call_llm = MagicMock(return_value=(LLM_RESPONSE, 200))

        result = s.synthesize(
            schema=SAMPLE_SCHEMA,
            seed_samples=SAMPLE_SEEDS,
            output_path=str(output_path),
            target_count=3,
            output_format="jsonl",
            resume=True,
        )

        assert result.success
        assert result.generated_count == 3  # 1 existing + 2 new
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

    def test_resume_no_existing_file(self, tmp_path):
        """Test resume when no file exists behaves like normal run."""
        output_path = tmp_path / "out.json"

        cfg = SynthesisConfig(target_count=2, batch_size=2)
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        s._call_llm = MagicMock(return_value=(LLM_RESPONSE, 200))

        result = s.synthesize(
            schema=SAMPLE_SCHEMA,
            seed_samples=SAMPLE_SEEDS,
            output_path=str(output_path),
            target_count=2,
            resume=True,
        )

        assert result.success
        assert result.generated_count == 2

    def test_estimate_cost(self):
        s = DataSynthesizer()
        cost = s._estimate_cost(10000)
        assert cost > 0

    def test_synthesize_populates_stats(self, tmp_path):
        """Test that synthesize populates result.stats."""
        cfg = SynthesisConfig(target_count=2, batch_size=2)
        s = DataSynthesizer(cfg)
        s._client = MagicMock()
        s._provider = "anthropic"
        s._call_llm = MagicMock(return_value=(LLM_RESPONSE, 500))

        result = s.synthesize(
            schema=SAMPLE_SCHEMA,
            seed_samples=SAMPLE_SEEDS,
            output_path=str(tmp_path / "out.json"),
            target_count=2,
        )

        assert result.success
        assert result.stats is not None
        assert result.stats["total_samples"] == 2
        assert "instruction" in result.stats["fields"]
        assert "response" in result.stats["fields"]
        assert result.stats["fields"]["instruction"]["type"] == "text"
        assert result.stats["fields"]["instruction"]["count"] == 2


class TestComputeStats:
    """Tests for DataSynthesizer._compute_stats."""

    def _schema(self, fields):
        from datasynth.config import DataSchema
        return DataSchema.from_dict({"project_name": "test", "fields": fields})

    def test_text_fields(self):
        schema = self._schema([{"name": "q", "type": "text"}, {"name": "a", "type": "text"}])
        samples = [
            {"q": "hello", "a": "world"},
            {"q": "hi", "a": "there!"},
        ]
        stats = DataSynthesizer._compute_stats(samples, schema)

        assert stats["total_samples"] == 2
        q = stats["fields"]["q"]
        assert q["type"] == "text"
        assert q["count"] == 2
        assert q["missing"] == 0
        assert q["avg_length"] == 3.5  # (5+2)/2
        assert q["min_length"] == 2
        assert q["max_length"] == 5

    def test_numeric_fields_with_distribution(self):
        schema = self._schema([{"name": "score", "type": "int"}])
        samples = [{"score": 3}, {"score": 5}, {"score": 3}, {"score": 4}]
        stats = DataSynthesizer._compute_stats(samples, schema)

        s = stats["fields"]["score"]
        assert s["type"] == "numeric"
        assert s["min"] == 3
        assert s["max"] == 5
        assert s["avg"] == 3.75
        assert s["distribution"] == {"3": 2, "4": 1, "5": 1}

    def test_list_fields(self):
        schema = self._schema([{"name": "tags", "type": "list"}])
        samples = [{"tags": ["a", "b"]}, {"tags": ["x"]}]
        stats = DataSynthesizer._compute_stats(samples, schema)

        t = stats["fields"]["tags"]
        assert t["type"] == "list"
        assert t["avg_items"] == 1.5
        assert t["min_items"] == 1
        assert t["max_items"] == 2

    def test_missing_values(self):
        schema = self._schema([{"name": "q", "type": "text"}, {"name": "note", "type": "text"}])
        samples = [{"q": "hi"}, {"q": "hello", "note": "ok"}]
        stats = DataSynthesizer._compute_stats(samples, schema)

        assert stats["fields"]["note"]["count"] == 1
        assert stats["fields"]["note"]["missing"] == 1

    def test_empty_samples(self):
        schema = self._schema([{"name": "q", "type": "text"}])
        stats = DataSynthesizer._compute_stats([], schema)
        assert stats["total_samples"] == 0
        assert stats["fields"]["q"]["count"] == 0

    def test_mixed_types(self):
        schema = self._schema([{"name": "val", "type": "text"}])
        samples = [{"val": "string"}, {"val": 123}]
        stats = DataSynthesizer._compute_stats(samples, schema)
        assert stats["fields"]["val"]["type"] == "mixed"


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
