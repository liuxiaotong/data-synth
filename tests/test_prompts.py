"""Tests for prompt generation and parsing."""

import pytest

from datasynth.config import DataSchema
from datasynth.prompts import (
    build_synthesis_prompt,
    extract_json_array,
    extract_json_objects,
    format_seed_samples,
    parse_generated_samples,
)


@pytest.fixture
def sample_schema():
    """Sample schema for testing."""
    return DataSchema.from_dict({
        "project_name": "测试项目",
        "fields": [
            {"name": "instruction", "display_name": "指令", "type": "text"},
            {"name": "response", "display_name": "回复", "type": "text"},
        ],
        "scoring_rubric": [
            {"score": 1, "label": "差"},
            {"score": 3, "label": "好"},
        ],
    })


@pytest.fixture
def sample_seeds():
    """Sample seed data."""
    return [
        {"instruction": "什么是 AI？", "response": "AI 是人工智能的缩写..."},
        {"instruction": "解释机器学习", "response": "机器学习是..."},
    ]


class TestBuildSynthesisPrompt:
    """Tests for prompt building."""

    def test_basic_prompt(self, sample_schema, sample_seeds):
        """Test basic prompt generation."""
        prompt = build_synthesis_prompt(
            schema=sample_schema,
            seed_samples=sample_seeds,
            count=5,
        )

        assert "测试项目" in prompt
        assert "5" in prompt
        assert "instruction" in prompt
        assert "response" in prompt
        assert "什么是 AI" in prompt

    def test_with_guidelines(self, sample_schema, sample_seeds):
        """Test prompt with guidelines."""
        prompt = build_synthesis_prompt(
            schema=sample_schema,
            seed_samples=sample_seeds,
            count=5,
            guidelines="请确保回复简洁明了",
        )

        assert "简洁明了" in prompt

    def test_diversity_low(self, sample_schema, sample_seeds):
        """Test low diversity instructions."""
        prompt = build_synthesis_prompt(
            schema=sample_schema,
            seed_samples=sample_seeds,
            count=5,
            diversity_factor=0.1,
        )

        assert "非常相似" in prompt

    def test_diversity_high(self, sample_schema, sample_seeds):
        """Test high diversity instructions."""
        prompt = build_synthesis_prompt(
            schema=sample_schema,
            seed_samples=sample_seeds,
            count=5,
            diversity_factor=0.9,
        )

        assert "较大变化" in prompt


class TestFormatSeedSamples:
    """Tests for seed sample formatting."""

    def test_format_samples(self, sample_schema, sample_seeds):
        """Test sample formatting."""
        formatted = format_seed_samples(sample_seeds, sample_schema)

        assert "样例 1" in formatted
        assert "样例 2" in formatted
        assert "什么是 AI" in formatted
        assert "```json" in formatted


class TestExtractJsonArray:
    """Tests for JSON array extraction."""

    def test_code_block(self):
        """Test extraction from code block."""
        text = '''Here is the data:
```json
[{"a": 1}, {"a": 2}]
```
'''
        result = extract_json_array(text)
        assert result == '[{"a": 1}, {"a": 2}]'

    def test_direct_array(self):
        """Test extraction of direct array."""
        text = 'Some text [{"a": 1}, {"a": 2}] more text'
        result = extract_json_array(text)
        assert '[{"a": 1}, {"a": 2}]' in result

    def test_no_array(self):
        """Test when no array present."""
        text = "Just some text without JSON"
        result = extract_json_array(text)
        assert result is None


class TestExtractJsonObjects:
    """Tests for JSON object extraction."""

    def test_multiple_objects(self):
        """Test extracting multiple objects."""
        text = '{"a": 1} some text {"b": 2}'
        objects = extract_json_objects(text)

        assert len(objects) == 2
        assert objects[0]["a"] == 1
        assert objects[1]["b"] == 2

    def test_invalid_json(self):
        """Test handling invalid JSON."""
        text = '{invalid} {"valid": true}'
        objects = extract_json_objects(text)

        assert len(objects) == 1
        assert objects[0]["valid"] is True


class TestParseGeneratedSamples:
    """Tests for parsing generated samples."""

    def test_parse_array(self, sample_schema):
        """Test parsing JSON array."""
        response = '''
```json
[
  {"instruction": "问题1", "response": "回答1"},
  {"instruction": "问题2", "response": "回答2"}
]
```
'''
        samples = parse_generated_samples(response, sample_schema)

        assert len(samples) == 2
        assert samples[0]["instruction"] == "问题1"

    def test_parse_with_extra_text(self, sample_schema):
        """Test parsing with surrounding text."""
        response = '''
Here are the generated samples:

[{"instruction": "问题", "response": "回答"}]

Hope this helps!
'''
        samples = parse_generated_samples(response, sample_schema)

        assert len(samples) == 1

    def test_filter_invalid_samples(self, sample_schema):
        """Test filtering invalid samples."""
        response = '''
[
  {"instruction": "valid", "response": "valid"},
  {"unrelated_field": "invalid"}
]
'''
        samples = parse_generated_samples(response, sample_schema)

        assert len(samples) == 1
        assert samples[0]["instruction"] == "valid"
