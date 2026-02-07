"""Prompt templates for data synthesis."""

import json
import re
from typing import Any, Dict, List

from datasynth.config import DataSchema


def build_synthesis_prompt(
    schema: DataSchema,
    seed_samples: List[Dict[str, Any]],
    count: int,
    guidelines: str | None = None,
    diversity_factor: float = 0.5,
) -> str:
    """Build prompt for data synthesis.

    Args:
        schema: Data schema definition
        seed_samples: Seed examples to learn from
        count: Number of samples to generate
        guidelines: Additional generation guidelines
        diversity_factor: 0.0 = very similar to seeds, 1.0 = very different

    Returns:
        Prompt string for LLM
    """
    # Schema description
    schema_desc = schema.to_prompt_description()

    # Format seed samples
    seeds_text = format_seed_samples(seed_samples, schema)

    # Diversity instructions
    if diversity_factor < 0.3:
        diversity_inst = "生成的数据应该与种子样例非常相似，保持相同的风格和难度。"
    elif diversity_factor < 0.7:
        diversity_inst = "生成的数据应该有适度变化，保持核心特征但引入新的场景和内容。"
    else:
        diversity_inst = "生成的数据应该有较大变化，探索不同的场景、风格和难度级别。"

    # Build prompt
    prompt_parts = [
        "# 数据合成任务",
        "",
        "你是一个专业的数据生成专家。请根据以下 Schema 和种子样例，生成新的高质量数据。",
        "",
        "## 数据 Schema",
        "",
        schema_desc,
        "",
        "## 种子样例",
        "",
        "以下是一些参考样例，请学习其格式、风格和质量标准：",
        "",
        seeds_text,
        "",
        "## 生成要求",
        "",
        f"1. 请生成 **{count}** 条新数据",
        f"2. {diversity_inst}",
        "3. 确保每条数据都符合 Schema 定义",
        "4. 保持数据的真实性和合理性",
        "5. 避免与种子样例重复",
        "",
    ]

    if guidelines:
        prompt_parts.extend(
            [
                "## 额外指南",
                "",
                guidelines[:2000],  # Truncate if too long
                "",
            ]
        )

    prompt_parts.extend(
        [
            "## 输出格式",
            "",
            "请以 JSON 数组格式输出，每个元素是一条数据：",
            "",
            "```json",
            "[",
            "  {",
        ]
    )

    # Add field placeholders
    field_examples = []
    for f in schema.fields:
        if f.name not in ["id", "metadata"]:
            field_examples.append(f'    "{f.name}": "..."')

    prompt_parts.append(",\n".join(field_examples))
    prompt_parts.extend(
        [
            "  },",
            "  ...",
            "]",
            "```",
            "",
            "现在请生成数据：",
        ]
    )

    return "\n".join(prompt_parts)


def format_seed_samples(
    samples: List[Dict[str, Any]],
    schema: DataSchema,
) -> str:
    """Format seed samples for prompt."""
    lines = []

    for i, sample in enumerate(samples, 1):
        lines.append(f"### 样例 {i}")
        lines.append("")
        lines.append("```json")
        # Only include schema fields
        filtered = {}
        field_names = {f.name for f in schema.fields}
        for key, value in sample.items():
            if key in field_names or not field_names:
                filtered[key] = value
        lines.append(json.dumps(filtered, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def parse_generated_samples(
    response_text: str,
    schema: DataSchema,
) -> List[Dict[str, Any]]:
    """Parse generated samples from LLM response.

    Handles various response formats:
    - Direct JSON array
    - JSON in code blocks
    - Mixed text with JSON
    """
    samples = []

    # Try to extract JSON array from response
    json_text = extract_json_array(response_text)

    if json_text:
        try:
            parsed = json.loads(json_text)
            if isinstance(parsed, list):
                samples = parsed
            elif isinstance(parsed, dict) and "samples" in parsed:
                samples = parsed["samples"]
        except json.JSONDecodeError:
            pass

    # If direct parsing failed, try to find individual JSON objects
    if not samples:
        samples = extract_json_objects(response_text)

    # Validate and filter samples
    valid_samples = []
    field_names = {f.name for f in schema.fields if f.name not in ["id", "metadata"]}

    for sample in samples:
        if isinstance(sample, dict):
            # Check if sample has required fields
            if field_names:
                has_fields = any(key in sample for key in field_names)
                if has_fields:
                    valid_samples.append(sample)
            else:
                valid_samples.append(sample)

    return valid_samples


def extract_json_array(text: str) -> str | None:
    """Extract JSON array from text."""
    # Try code block first
    code_block_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if code_block_match:
        return code_block_match.group(1)

    # Try direct JSON array
    array_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", text)
    if array_match:
        return array_match.group(0)

    return None


def extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """Extract individual JSON objects from text."""
    objects = []

    # Find all JSON object patterns
    pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"

    for match in re.finditer(pattern, text):
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            continue

    return objects


# Specialized prompts for different data types

PROMPT_TEMPLATES = {
    "instruction_response": """
# 指令-回复数据生成

请根据以下要求生成指令-回复对：

{schema_description}

## 种子样例
{seed_samples}

## 生成要求
- 生成 {count} 条数据
- 指令应该清晰、具体
- 回复应该准确、完整
- 保持多样性：涵盖不同领域和难度

## 输出格式
```json
[
  {{"instruction": "...", "response": "..."}},
  ...
]
```
""",
    "preference": """
# 偏好数据生成

请根据以下要求生成偏好对比数据：

{schema_description}

## 种子样例
{seed_samples}

## 生成要求
- 生成 {count} 条数据
- chosen 回复应该明显优于 rejected
- 质量差异应该符合评分标准
- 涵盖不同的错误类型和质量问题

## 输出格式
```json
[
  {{"prompt": "...", "chosen": "...", "rejected": "..."}},
  ...
]
```
""",
    "multi_turn": """
# 多轮对话数据生成

请根据以下要求生成多轮对话：

{schema_description}

## 种子样例
{seed_samples}

## 生成要求
- 生成 {count} 条对话
- 每条对话包含 2-5 轮交互
- 对话应该自然、连贯
- 涵盖不同的对话场景

## 输出格式
```json
[
  {{
    "conversation": [
      {{"role": "user", "content": "..."}},
      {{"role": "assistant", "content": "..."}},
      ...
    ]
  }},
  ...
]
```
""",
}


def get_specialized_prompt(
    data_type: str,
    schema: DataSchema,
    seed_samples: List[Dict[str, Any]],
    count: int,
) -> str:
    """Get specialized prompt for specific data types."""
    template = PROMPT_TEMPLATES.get(data_type)

    if not template:
        # Fall back to generic prompt
        return build_synthesis_prompt(schema, seed_samples, count)

    return template.format(
        schema_description=schema.to_prompt_description(),
        seed_samples=format_seed_samples(seed_samples, schema),
        count=count,
    )
