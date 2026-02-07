"""Tests for configuration."""

from datasynth.config import DataSchema, SchemaField, SynthesisConfig


class TestSynthesisConfig:
    """Tests for SynthesisConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SynthesisConfig()
        assert config.target_count == 100
        assert config.batch_size == 5
        assert config.temperature == 0.8

    def test_estimate_cost(self):
        """Test cost estimation."""
        config = SynthesisConfig(target_count=100)
        estimate = config.estimate_cost()

        assert estimate["target_count"] == 100
        assert estimate["estimated_batches"] == 20
        assert estimate["estimated_cost_usd"] > 0

    def test_estimate_cost_with_override(self):
        """Test cost estimation with count override."""
        config = SynthesisConfig(target_count=100)
        estimate = config.estimate_cost(target_count=500)

        assert estimate["target_count"] == 500
        assert estimate["estimated_batches"] == 100


class TestDataSchema:
    """Tests for DataSchema."""

    def test_from_dict(self):
        """Test schema parsing from dict."""
        data = {
            "project_name": "Test Project",
            "description": "A test project",
            "fields": [
                {"name": "instruction", "type": "text"},
                {"name": "response", "type": "text"},
            ],
            "scoring_rubric": [
                {"score": 1, "label": "Bad"},
                {"score": 2, "label": "Good"},
            ],
        }

        schema = DataSchema.from_dict(data)

        assert schema.project_name == "Test Project"
        assert len(schema.fields) == 2
        assert schema.fields[0].name == "instruction"
        assert len(schema.scoring_rubric) == 2

    def test_to_prompt_description(self):
        """Test schema to prompt conversion."""
        data = {
            "project_name": "Test",
            "fields": [
                {"name": "instruction", "display_name": "指令", "type": "text"},
            ],
        }

        schema = DataSchema.from_dict(data)
        desc = schema.to_prompt_description()

        assert "Test" in desc
        assert "instruction" in desc
        assert "指令" in desc


class TestSchemaField:
    """Tests for SchemaField."""

    def test_from_dict_minimal(self):
        """Test field parsing with minimal data."""
        data = {"name": "test_field"}
        field = SchemaField.from_dict(data)

        assert field.name == "test_field"
        assert field.type == "text"
        assert field.required is True

    def test_from_dict_full(self):
        """Test field parsing with full data."""
        data = {
            "name": "instruction",
            "display_name": "指令",
            "type": "text",
            "description": "用户输入的指令",
            "required": True,
            "constraints": {"max_length": 1000},
        }

        field = SchemaField.from_dict(data)

        assert field.name == "instruction"
        assert field.display_name == "指令"
        assert field.description == "用户输入的指令"
        assert field.constraints["max_length"] == 1000
