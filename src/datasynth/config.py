"""Synthesis configuration."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SynthesisConfig:
    """Configuration for data synthesis.

    Attributes:
        target_count: Number of samples to generate
        model: LLM model to use (claude-3-5-sonnet, gpt-4o, etc.)
        provider: LLM provider (anthropic, openai)
        temperature: Sampling temperature (0.0-1.0)
        diversity_factor: How much to vary from seed examples (0.0-1.0)
        batch_size: Number of samples per API call
        max_retries: Maximum retries on failure
        validate: Whether to validate generated samples
        seed_sample_count: Number of seed samples to include in prompt
    """

    target_count: int = 100
    model: str = "claude-sonnet-4-20250514"
    provider: str = "anthropic"
    temperature: float = 0.8
    diversity_factor: float = 0.5
    batch_size: int = 5
    max_retries: int = 3
    retry_delay: float = 2.0  # seconds between retries
    validate: bool = True
    seed_sample_count: int = 3

    # Cost tracking
    input_token_cost: float = 0.003  # per 1K tokens
    output_token_cost: float = 0.015  # per 1K tokens

    # Generation constraints
    max_tokens_per_sample: int = 2000

    def estimate_cost(self, target_count: Optional[int] = None) -> Dict[str, Any]:
        """Estimate generation cost.

        Returns:
            Dictionary with cost breakdown
        """
        count = target_count or self.target_count
        batches = (count + self.batch_size - 1) // self.batch_size

        # Rough estimates
        avg_input_tokens = 2000  # prompt + seed samples
        avg_output_tokens = self.batch_size * 500  # generated samples

        total_input = batches * avg_input_tokens
        total_output = batches * avg_output_tokens

        input_cost = (total_input / 1000) * self.input_token_cost
        output_cost = (total_output / 1000) * self.output_token_cost

        return {
            "target_count": count,
            "estimated_batches": batches,
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "estimated_cost_usd": round(input_cost + output_cost, 2),
            "model": self.model,
            "provider": self.provider,
        }


@dataclass
class SchemaField:
    """A field in the data schema."""

    name: str
    display_name: str = ""
    type: str = "text"
    description: str = ""
    required: bool = True
    constraints: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaField":
        return cls(
            name=data.get("name", ""),
            display_name=data.get("display_name", data.get("name", "")),
            type=data.get("type", "text"),
            description=data.get("description", ""),
            required=data.get("required", True),
            constraints=data.get("constraints", {}),
        )


@dataclass
class DataSchema:
    """Schema for generated data."""

    project_name: str = ""
    description: str = ""
    fields: List[SchemaField] = field(default_factory=list)
    scoring_rubric: List[Dict[str, Any]] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataSchema":
        fields = [SchemaField.from_dict(f) for f in data.get("fields", [])]
        return cls(
            project_name=data.get("project_name", ""),
            description=data.get("description", ""),
            fields=fields,
            scoring_rubric=data.get("scoring_rubric", []),
            constraints=data.get("constraints", {}),
        )

    def to_prompt_description(self) -> str:
        """Convert schema to natural language for prompting."""
        lines = [f"项目: {self.project_name}"]
        if self.description:
            lines.append(f"描述: {self.description}")

        lines.append("\n字段定义:")
        for f in self.fields:
            desc = f.description or f.display_name or f.name
            lines.append(f"- {f.name} ({f.type}): {desc}")

        if self.scoring_rubric:
            lines.append("\n评分标准:")
            for r in self.scoring_rubric:
                lines.append(f"- {r.get('score')}: {r.get('label')} - {r.get('description', '')}")

        return "\n".join(lines)
