"""Tests for CLI commands."""

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from datasynth.cli import main
from datasynth.synthesizer import SynthesisResult


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def datarecipe_dir(tmp_path):
    """Create a mock DataRecipe analysis directory."""
    schema_dir = tmp_path / "04_复刻指南"
    schema_dir.mkdir()
    (schema_dir / "DATA_SCHEMA.json").write_text(
        json.dumps(
            {
                "project_name": "测试",
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


class TestVersion:
    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.4.0" in result.output


class TestEstimate:
    def test_default(self, runner):
        result = runner.invoke(main, ["estimate"])
        assert result.exit_code == 0
        assert "成本估算" in result.output
        assert "目标数量: 100" in result.output

    def test_custom_count(self, runner):
        result = runner.invoke(main, ["estimate", "-n", "500"])
        assert result.exit_code == 0
        assert "目标数量: 500" in result.output

    def test_custom_model(self, runner):
        result = runner.invoke(main, ["estimate", "-m", "gpt-4o"])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output


class TestGenerate:
    def test_dry_run(self, runner, datarecipe_dir):
        result = runner.invoke(
            main, ["generate", str(datarecipe_dir), "--dry-run"]
        )
        assert result.exit_code == 0
        assert "成本估算" in result.output
        assert "目标数量: 100" in result.output

    def test_dry_run_custom_count(self, runner, datarecipe_dir):
        result = runner.invoke(
            main, ["generate", str(datarecipe_dir), "-n", "200", "--dry-run"]
        )
        assert result.exit_code == 0
        assert "目标数量: 200" in result.output

    def test_generate_with_retry_options(self, runner, datarecipe_dir):
        mock_result = SynthesisResult(
            success=True,
            output_path="x",
            generated_count=5,
            failed_count=0,
            total_tokens=1000,
            estimated_cost=0.01,
            duration_seconds=1.0,
        )

        with patch("datasynth.cli.DataSynthesizer") as MockSynth:
            MockSynth.return_value.synthesize_from_datarecipe.return_value = mock_result
            result = runner.invoke(
                main,
                [
                    "generate",
                    str(datarecipe_dir),
                    "-n",
                    "5",
                    "--max-retries",
                    "5",
                    "--retry-delay",
                    "0.5",
                ],
            )

        assert result.exit_code == 0
        # Verify config was created with retry options
        cfg = MockSynth.call_args[0][0] if MockSynth.call_args[0] else MockSynth.call_args[1].get("config")
        if cfg is None:
            cfg = MockSynth.call_args[0][0]
        assert cfg.max_retries == 5
        assert cfg.retry_delay == 0.5

    def test_generate_success(self, runner, datarecipe_dir):
        mock_result = SynthesisResult(
            success=True,
            output_path=str(datarecipe_dir / "11_合成数据" / "synthetic.json"),
            generated_count=10,
            failed_count=0,
            total_tokens=5000,
            estimated_cost=0.05,
            duration_seconds=2.5,
        )

        with patch(
            "datasynth.cli.DataSynthesizer"
        ) as MockSynth:
            MockSynth.return_value.synthesize_from_datarecipe.return_value = mock_result
            result = runner.invoke(
                main, ["generate", str(datarecipe_dir), "-n", "10"]
            )

        assert result.exit_code == 0
        assert "生成成功" in result.output
        assert "生成数量: 10" in result.output

    def test_generate_with_post_hook(self, runner, datarecipe_dir):
        mock_result = SynthesisResult(
            success=True,
            output_path=str(datarecipe_dir / "out.json"),
            generated_count=5,
            failed_count=0,
            total_tokens=1000,
            estimated_cost=0.01,
            duration_seconds=1.0,
        )

        with patch("datasynth.cli.DataSynthesizer") as MockSynth, patch(
            "datasynth.cli.subprocess.run"
        ) as mock_run:
            MockSynth.return_value.synthesize_from_datarecipe.return_value = mock_result
            mock_run.return_value.returncode = 0
            result = runner.invoke(
                main,
                [
                    "generate",
                    str(datarecipe_dir),
                    "-n",
                    "5",
                    "--post-hook",
                    "echo {analysis_dir} {output_path} {count}",
                ],
            )

        assert result.exit_code == 0
        assert "post-hook" in result.output
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert str(datarecipe_dir) in cmd
        assert "5" in cmd

    def test_generate_post_hook_failure(self, runner, datarecipe_dir):
        mock_result = SynthesisResult(
            success=True,
            output_path="x",
            generated_count=5,
            failed_count=0,
            total_tokens=100,
            estimated_cost=0.01,
            duration_seconds=0.5,
        )

        with patch("datasynth.cli.DataSynthesizer") as MockSynth, patch(
            "datasynth.cli.subprocess.run"
        ) as mock_run:
            MockSynth.return_value.synthesize_from_datarecipe.return_value = mock_result
            mock_run.return_value.returncode = 1
            result = runner.invoke(
                main,
                [
                    "generate",
                    str(datarecipe_dir),
                    "-n",
                    "5",
                    "--post-hook",
                    "false",
                ],
            )

        assert result.exit_code == 0  # generate itself succeeds
        assert "post-hook 退出码: 1" in result.output

    def test_generate_no_post_hook_on_failure(self, runner, datarecipe_dir):
        """Post-hook should NOT run when generation fails."""
        mock_result = SynthesisResult(success=False, error="fail")

        with patch("datasynth.cli.DataSynthesizer") as MockSynth, patch(
            "datasynth.cli.subprocess.run"
        ) as mock_run:
            MockSynth.return_value.synthesize_from_datarecipe.return_value = mock_result
            result = runner.invoke(
                main,
                [
                    "generate",
                    str(datarecipe_dir),
                    "-n",
                    "5",
                    "--post-hook",
                    "echo should-not-run",
                ],
            )

        assert result.exit_code == 1
        mock_run.assert_not_called()

    def test_generate_failure(self, runner, datarecipe_dir):
        mock_result = SynthesisResult(success=False, error="API key not set")

        with patch(
            "datasynth.cli.DataSynthesizer"
        ) as MockSynth:
            MockSynth.return_value.synthesize_from_datarecipe.return_value = mock_result
            result = runner.invoke(
                main, ["generate", str(datarecipe_dir), "-n", "10"]
            )

        assert result.exit_code == 1
        assert "生成失败" in result.output

    def test_generate_with_data_type(self, runner, datarecipe_dir):
        mock_result = SynthesisResult(
            success=True,
            output_path="x",
            generated_count=5,
            failed_count=0,
            total_tokens=1000,
            estimated_cost=0.01,
            duration_seconds=1.0,
        )

        with patch("datasynth.cli.DataSynthesizer") as MockSynth:
            MockSynth.return_value.synthesize_from_datarecipe.return_value = mock_result
            result = runner.invoke(
                main,
                [
                    "generate",
                    str(datarecipe_dir),
                    "-n",
                    "5",
                    "--data-type",
                    "preference",
                ],
            )

        assert result.exit_code == 0
        cfg = MockSynth.call_args[0][0]
        assert cfg.data_type == "preference"

    def test_generate_with_resume(self, runner, datarecipe_dir):
        mock_result = SynthesisResult(
            success=True,
            output_path="x",
            generated_count=10,
            failed_count=0,
            total_tokens=2000,
            estimated_cost=0.02,
            duration_seconds=2.0,
        )

        with patch("datasynth.cli.DataSynthesizer") as MockSynth:
            MockSynth.return_value.synthesize_from_datarecipe.return_value = mock_result
            result = runner.invoke(
                main,
                [
                    "generate",
                    str(datarecipe_dir),
                    "-n",
                    "10",
                    "--resume",
                ],
            )

        assert result.exit_code == 0
        assert "增量续跑" in result.output
        # Verify resume=True was passed
        call_kwargs = MockSynth.return_value.synthesize_from_datarecipe.call_args[1]
        assert call_kwargs["resume"] is True

    def test_nonexistent_dir(self, runner):
        result = runner.invoke(main, ["generate", "/nonexistent/path"])
        assert result.exit_code != 0


class TestCreate:
    def test_create_success(self, runner, tmp_path):
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(
            json.dumps(
                {
                    "project_name": "Test",
                    "fields": [{"name": "text", "type": "text"}],
                }
            ),
            encoding="utf-8",
        )

        seeds_file = tmp_path / "seeds.json"
        seeds_file.write_text(
            json.dumps([{"text": "hello"}]),
            encoding="utf-8",
        )

        mock_result = SynthesisResult(
            success=True,
            output_path=str(tmp_path / "out.json"),
            generated_count=5,
        )

        with patch("datasynth.cli.DataSynthesizer") as MockSynth:
            MockSynth.return_value.synthesize.return_value = mock_result
            result = runner.invoke(
                main,
                [
                    "create",
                    str(schema_file),
                    str(seeds_file),
                    "-o",
                    str(tmp_path / "out.json"),
                    "-n",
                    "5",
                ],
            )

        assert result.exit_code == 0
        assert "生成成功" in result.output

    def test_create_with_data_field(self, runner, tmp_path):
        """Test seeds with 'data' sub-field."""
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps({"fields": [{"name": "q"}]}), encoding="utf-8")

        seeds_file = tmp_path / "seeds.json"
        seeds_file.write_text(
            json.dumps({"samples": [{"data": {"q": "hi"}}]}),
            encoding="utf-8",
        )

        mock_result = SynthesisResult(success=True, output_path="x", generated_count=1)

        with patch("datasynth.cli.DataSynthesizer") as MockSynth:
            MockSynth.return_value.synthesize.return_value = mock_result
            result = runner.invoke(
                main,
                ["create", str(schema_file), str(seeds_file), "-o", str(tmp_path / "o.json")],
            )

        assert result.exit_code == 0

    def test_create_empty_seeds(self, runner, tmp_path):
        schema_file = tmp_path / "schema.json"
        schema_file.write_text("{}", encoding="utf-8")

        seeds_file = tmp_path / "seeds.json"
        seeds_file.write_text("[]", encoding="utf-8")

        result = runner.invoke(
            main,
            ["create", str(schema_file), str(seeds_file), "-o", str(tmp_path / "o.json")],
        )
        assert result.exit_code == 1
        assert "种子数据为空" in result.output


    def test_create_with_resume(self, runner, tmp_path):
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(
            json.dumps({"project_name": "T", "fields": [{"name": "text", "type": "text"}]}),
            encoding="utf-8",
        )

        seeds_file = tmp_path / "seeds.json"
        seeds_file.write_text(json.dumps([{"text": "hello"}]), encoding="utf-8")

        mock_result = SynthesisResult(success=True, output_path=str(tmp_path / "o.json"), generated_count=5)

        with patch("datasynth.cli.DataSynthesizer") as MockSynth:
            MockSynth.return_value.synthesize.return_value = mock_result
            result = runner.invoke(
                main,
                ["create", str(schema_file), str(seeds_file), "-o", str(tmp_path / "o.json"), "--resume"],
            )

        assert result.exit_code == 0
        call_kwargs = MockSynth.return_value.synthesize.call_args[1]
        assert call_kwargs["resume"] is True


class TestStats:
    def test_generate_with_stats(self, runner, datarecipe_dir):
        output_dir = datarecipe_dir / "11_合成数据"
        output_dir.mkdir()
        mock_result = SynthesisResult(
            success=True,
            output_path=str(output_dir / "synthetic.json"),
            generated_count=5,
            failed_count=0,
            total_tokens=1000,
            estimated_cost=0.01,
            duration_seconds=1.0,
            stats={"total_samples": 5, "fields": {"instruction": {"count": 5}}},
        )

        with patch("datasynth.cli.DataSynthesizer") as MockSynth:
            MockSynth.return_value.synthesize_from_datarecipe.return_value = mock_result
            result = runner.invoke(
                main,
                ["generate", str(datarecipe_dir), "-n", "5", "--stats"],
            )

        assert result.exit_code == 0
        assert "统计报告" in result.output
        stats_path = datarecipe_dir / "11_合成数据" / "synthetic.stats.json"
        assert stats_path.exists()
        import json
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        assert stats["total_samples"] == 5

    def test_create_with_stats(self, runner, tmp_path):
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(
            json.dumps({"project_name": "T", "fields": [{"name": "text", "type": "text"}]}),
            encoding="utf-8",
        )

        seeds_file = tmp_path / "seeds.json"
        seeds_file.write_text(json.dumps([{"text": "hello"}]), encoding="utf-8")

        out = tmp_path / "out.json"
        mock_result = SynthesisResult(
            success=True,
            output_path=str(out),
            generated_count=3,
            stats={"total_samples": 3, "fields": {"text": {"count": 3}}},
        )

        with patch("datasynth.cli.DataSynthesizer") as MockSynth:
            MockSynth.return_value.synthesize.return_value = mock_result
            result = runner.invoke(
                main,
                ["create", str(schema_file), str(seeds_file), "-o", str(out), "-n", "3", "--stats"],
            )

        assert result.exit_code == 0
        assert "统计报告" in result.output
        stats_path = tmp_path / "out.stats.json"
        assert stats_path.exists()


class TestVerbose:
    def test_verbose_flag(self, runner, datarecipe_dir):
        mock_result = SynthesisResult(
            success=True, output_path="x", generated_count=2,
            failed_count=0, total_tokens=100, estimated_cost=0.01, duration_seconds=0.5,
        )

        with patch("datasynth.cli.DataSynthesizer") as MockSynth:
            MockSynth.return_value.synthesize_from_datarecipe.return_value = mock_result
            result = runner.invoke(
                main, ["-v", "generate", str(datarecipe_dir), "-n", "2"]
            )

        assert result.exit_code == 0
        assert "生成成功" in result.output


class TestPrepare:
    def test_prepare_stdout(self, runner, datarecipe_dir):
        result = runner.invoke(main, ["prepare", str(datarecipe_dir), "-n", "5"])
        assert result.exit_code == 0
        # Auto-detect instruction_response → specialized prompt
        assert "指令-回复" in result.output
        assert "预期生成 5 条数据" in result.output

    def test_prepare_to_file(self, runner, datarecipe_dir, tmp_path):
        out = tmp_path / "prompt.txt"
        result = runner.invoke(
            main, ["prepare", str(datarecipe_dir), "-o", str(out)]
        )
        assert result.exit_code == 0
        assert "Prompt 已保存" in result.output
        assert out.exists()

    def test_prepare_missing_schema(self, runner, tmp_path):
        result = runner.invoke(main, ["prepare", str(tmp_path)])
        assert result.exit_code == 1
        assert "Schema 未找到" in result.output

    def test_prepare_missing_samples(self, runner, tmp_path):
        schema_dir = tmp_path / "04_复刻指南"
        schema_dir.mkdir()
        (schema_dir / "DATA_SCHEMA.json").write_text("{}", encoding="utf-8")

        result = runner.invoke(main, ["prepare", str(tmp_path)])
        assert result.exit_code == 1
        assert "种子数据未找到" in result.output

    def test_prepare_with_data_type(self, runner, datarecipe_dir):
        result = runner.invoke(
            main, ["prepare", str(datarecipe_dir), "-n", "3", "--data-type", "preference"]
        )
        assert result.exit_code == 0
        assert "偏好对比" in result.output

    def test_prepare_with_guidelines(self, runner, datarecipe_dir):
        """prepare should load guidelines from 03_标注规范."""
        guidelines_dir = datarecipe_dir / "03_标注规范"
        guidelines_dir.mkdir()
        (guidelines_dir / "ANNOTATION_SPEC.md").write_text("自定义标注要求", encoding="utf-8")

        result = runner.invoke(main, ["prepare", str(datarecipe_dir), "-n", "3"])
        assert result.exit_code == 0
        assert "自定义标注要求" in result.output


class TestCreateDryRun:
    def test_create_dry_run(self, runner, tmp_path):
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(
            json.dumps({"project_name": "T", "fields": [{"name": "text", "type": "text"}]}),
            encoding="utf-8",
        )
        seeds_file = tmp_path / "seeds.json"
        seeds_file.write_text(json.dumps([{"text": "hello"}]), encoding="utf-8")

        result = runner.invoke(
            main,
            ["create", str(schema_file), str(seeds_file), "-o", str(tmp_path / "o.json"), "--dry-run"],
        )
        assert result.exit_code == 0
        assert "成本估算" in result.output
        assert "目标数量: 100" in result.output

    def test_create_post_hook(self, runner, tmp_path):
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(
            json.dumps({"project_name": "T", "fields": [{"name": "text", "type": "text"}]}),
            encoding="utf-8",
        )
        seeds_file = tmp_path / "seeds.json"
        seeds_file.write_text(json.dumps([{"text": "hello"}]), encoding="utf-8")

        mock_result = SynthesisResult(
            success=True, output_path=str(tmp_path / "o.json"), generated_count=5,
        )

        with patch("datasynth.cli.DataSynthesizer") as MockSynth, patch(
            "datasynth.cli.subprocess.run"
        ) as mock_run:
            MockSynth.return_value.synthesize.return_value = mock_result
            mock_run.return_value.returncode = 0
            result = runner.invoke(
                main,
                [
                    "create", str(schema_file), str(seeds_file),
                    "-o", str(tmp_path / "o.json"), "-n", "5",
                    "--post-hook", "echo {output_path} {count}",
                ],
            )

        assert result.exit_code == 0
        assert "post-hook" in result.output
        mock_run.assert_called_once()


class TestDryRunSchemaInfo:
    def test_generate_dry_run_shows_schema(self, runner, datarecipe_dir):
        result = runner.invoke(
            main, ["generate", str(datarecipe_dir), "--dry-run"]
        )
        assert result.exit_code == 0
        assert "Schema 信息" in result.output
        assert "instruction" in result.output
        assert "instruction_response" in result.output

    def test_generate_dry_run_shows_guidelines(self, runner, datarecipe_dir):
        guidelines_dir = datarecipe_dir / "03_标注规范"
        guidelines_dir.mkdir()
        (guidelines_dir / "ANNOTATION_SPEC.md").write_text("spec", encoding="utf-8")

        result = runner.invoke(
            main, ["generate", str(datarecipe_dir), "--dry-run"]
        )
        assert result.exit_code == 0
        assert "标注规范: ✓" in result.output


class TestConfigFile:
    def test_generate_with_config(self, runner, datarecipe_dir, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps({"target_count": 50, "model": "gpt-4o", "batch_size": 10}),
            encoding="utf-8",
        )

        result = runner.invoke(
            main, ["generate", str(datarecipe_dir), "--dry-run", "--config", str(config_file)]
        )
        assert result.exit_code == 0
        assert "gpt-4o" in result.output

    def test_cli_overrides_config(self, runner, datarecipe_dir, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps({"model": "gpt-4o"}),
            encoding="utf-8",
        )

        # CLI -m should override config
        result = runner.invoke(
            main, ["generate", str(datarecipe_dir), "--dry-run", "--config", str(config_file), "-m", "claude-haiku-3-5"]
        )
        assert result.exit_code == 0
        assert "claude-haiku-3-5" in result.output


class TestValidate:
    def test_validate_all_valid(self, runner, tmp_path):
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(
            json.dumps({"fields": [{"name": "q", "type": "text"}, {"name": "a", "type": "text"}]}),
            encoding="utf-8",
        )
        data_file = tmp_path / "data.json"
        data_file.write_text(
            json.dumps([{"q": "hello", "a": "world"}, {"q": "hi", "a": "there"}]),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["validate", str(data_file), str(schema_file)])
        assert result.exit_code == 0
        assert "合规: 2" in result.output
        assert "全部通过" in result.output

    def test_validate_with_errors(self, runner, tmp_path):
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(
            json.dumps({"fields": [
                {"name": "score", "type": "int", "constraints": {"range": [1, 5]}},
            ]}),
            encoding="utf-8",
        )
        data_file = tmp_path / "data.json"
        data_file.write_text(
            json.dumps([{"score": 3}, {"score": 10}, {"score": "bad"}]),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["validate", str(data_file), str(schema_file)])
        assert result.exit_code == 1
        assert "合规: 1" in result.output
        assert "不合规: 2" in result.output

    def test_validate_jsonl(self, runner, tmp_path):
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(
            json.dumps({"fields": [{"name": "text", "type": "text"}]}),
            encoding="utf-8",
        )
        data_file = tmp_path / "data.jsonl"
        data_file.write_text(
            '{"text": "hello"}\n{"text": "world"}\n',
            encoding="utf-8",
        )

        result = runner.invoke(main, ["validate", str(data_file), str(schema_file)])
        assert result.exit_code == 0
        assert "合规: 2" in result.output

    def test_validate_samples_format(self, runner, tmp_path):
        """Validate should handle {samples: [{data: ...}]} format."""
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(
            json.dumps({"fields": [{"name": "q", "type": "text"}]}),
            encoding="utf-8",
        )
        data_file = tmp_path / "data.json"
        data_file.write_text(
            json.dumps({"samples": [{"data": {"q": "hi"}}, {"data": {"q": "hello"}}]}),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["validate", str(data_file), str(schema_file)])
        assert result.exit_code == 0
        assert "合规: 2" in result.output

    def test_validate_empty_data(self, runner, tmp_path):
        schema_file = tmp_path / "schema.json"
        schema_file.write_text("{}", encoding="utf-8")
        data_file = tmp_path / "data.json"
        data_file.write_text("[]", encoding="utf-8")

        result = runner.invoke(main, ["validate", str(data_file), str(schema_file)])
        assert result.exit_code == 1
        assert "数据文件为空" in result.output


class TestInit:
    def test_init_creates_files(self, runner, tmp_path):
        result = runner.invoke(main, ["init", "-o", str(tmp_path / "project")])
        assert result.exit_code == 0
        assert (tmp_path / "project" / "datasynth.config.json").exists()
        assert (tmp_path / "project" / "schema.json").exists()
        assert (tmp_path / "project" / "seeds.json").exists()
        assert "已创建" in result.output
        assert "快速开始" in result.output

    def test_init_skips_existing(self, runner, tmp_path):
        (tmp_path / "datasynth.config.json").write_text("{}", encoding="utf-8")
        result = runner.invoke(main, ["init", "-o", str(tmp_path)])
        assert result.exit_code == 0
        assert "跳过" in result.output

    def test_init_default_directory(self, runner, tmp_path):
        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = runner.invoke(main, ["init"])
            assert result.exit_code == 0
            assert (tmp_path / "schema.json").exists()
        finally:
            os.chdir(old_cwd)
