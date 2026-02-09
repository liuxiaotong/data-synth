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
        assert "0.1.0" in result.output


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


class TestPrepare:
    def test_prepare_stdout(self, runner, datarecipe_dir):
        result = runner.invoke(main, ["prepare", str(datarecipe_dir), "-n", "5"])
        assert result.exit_code == 0
        assert "数据合成任务" in result.output
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
