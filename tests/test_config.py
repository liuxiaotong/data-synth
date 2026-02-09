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


class TestDetectDataType:
    """Tests for DataSchema.detect_data_type."""

    def _schema(self, fields):
        return DataSchema.from_dict({"project_name": "test", "fields": fields})

    def test_instruction_response(self):
        schema = self._schema([
            {"name": "instruction", "type": "text"},
            {"name": "response", "type": "text"},
        ])
        assert schema.detect_data_type() == "instruction_response"

    def test_input_output(self):
        schema = self._schema([
            {"name": "input", "type": "text"},
            {"name": "output", "type": "text"},
        ])
        assert schema.detect_data_type() == "instruction_response"

    def test_preference(self):
        schema = self._schema([
            {"name": "prompt", "type": "text"},
            {"name": "chosen", "type": "text"},
            {"name": "rejected", "type": "text"},
        ])
        assert schema.detect_data_type() == "preference"

    def test_multi_turn(self):
        schema = self._schema([
            {"name": "conversation", "type": "list"},
        ])
        assert schema.detect_data_type() == "multi_turn"

    def test_unknown(self):
        schema = self._schema([
            {"name": "foo", "type": "text"},
            {"name": "bar", "type": "int"},
        ])
        assert schema.detect_data_type() is None


class TestSchemaValidation:
    """Tests for DataSchema.validate_sample."""

    def _schema(self, fields):
        return DataSchema.from_dict({"project_name": "test", "fields": fields})

    def test_valid_sample(self):
        schema = self._schema([
            {"name": "q", "type": "text"},
            {"name": "a", "type": "text"},
        ])
        assert schema.validate_sample({"q": "hello", "a": "world"}) == []

    def test_missing_required_field(self):
        schema = self._schema([
            {"name": "q", "type": "text", "required": True},
            {"name": "a", "type": "text", "required": True},
        ])
        errs = schema.validate_sample({"q": "hello"})
        assert len(errs) == 1
        assert "missing required field: a" in errs[0]

    def test_optional_field_can_be_absent(self):
        schema = self._schema([
            {"name": "q", "type": "text"},
            {"name": "note", "type": "text", "required": False},
        ])
        assert schema.validate_sample({"q": "hello"}) == []

    def test_type_check_int(self):
        schema = self._schema([
            {"name": "score", "type": "int"},
        ])
        assert schema.validate_sample({"score": 5}) == []
        errs = schema.validate_sample({"score": "five"})
        assert any("expected int" in e for e in errs)

    def test_type_check_float(self):
        schema = self._schema([
            {"name": "weight", "type": "float"},
        ])
        assert schema.validate_sample({"weight": 3.14}) == []
        assert schema.validate_sample({"weight": 3}) == []  # int is also valid for float
        errs = schema.validate_sample({"weight": "heavy"})
        assert any("expected number" in e for e in errs)

    def test_type_check_bool(self):
        schema = self._schema([
            {"name": "active", "type": "bool"},
        ])
        assert schema.validate_sample({"active": True}) == []
        errs = schema.validate_sample({"active": 1})
        assert any("expected bool" in e for e in errs)

    def test_type_check_list(self):
        schema = self._schema([
            {"name": "tags", "type": "list"},
        ])
        assert schema.validate_sample({"tags": ["a", "b"]}) == []
        errs = schema.validate_sample({"tags": "a,b"})
        assert any("expected list" in e for e in errs)

    def test_range_constraint(self):
        schema = self._schema([
            {"name": "score", "type": "int", "constraints": {"range": [1, 5]}},
        ])
        assert schema.validate_sample({"score": 3}) == []
        errs = schema.validate_sample({"score": 0})
        assert any("out of range" in e for e in errs)
        errs = schema.validate_sample({"score": 6})
        assert any("out of range" in e for e in errs)

    def test_enum_constraint(self):
        schema = self._schema([
            {"name": "level", "type": "text", "constraints": {"enum": ["low", "mid", "high"]}},
        ])
        assert schema.validate_sample({"level": "low"}) == []
        errs = schema.validate_sample({"level": "ultra"})
        assert any("not in allowed values" in e for e in errs)

    def test_min_length_constraint(self):
        schema = self._schema([
            {"name": "desc", "type": "text", "constraints": {"min_length": 5}},
        ])
        assert schema.validate_sample({"desc": "hello"}) == []
        errs = schema.validate_sample({"desc": "hi"})
        assert any("min_length" in e for e in errs)

    def test_max_length_constraint(self):
        schema = self._schema([
            {"name": "code", "type": "text", "constraints": {"max_length": 3}},
        ])
        assert schema.validate_sample({"code": "abc"}) == []
        errs = schema.validate_sample({"code": "abcdef"})
        assert any("max_length" in e for e in errs)

    def test_multiple_errors(self):
        schema = self._schema([
            {"name": "q", "type": "text", "required": True},
            {"name": "score", "type": "int", "constraints": {"range": [1, 5]}},
        ])
        errs = schema.validate_sample({"score": 10})
        assert len(errs) == 2  # missing q + score out of range

    def test_bool_not_treated_as_int(self):
        """Booleans should NOT pass int validation."""
        schema = self._schema([
            {"name": "count", "type": "int"},
        ])
        errs = schema.validate_sample({"count": True})
        assert any("expected int" in e for e in errs)
