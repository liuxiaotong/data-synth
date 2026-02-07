"""Core data synthesizer."""

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from datasynth.config import DataSchema, SynthesisConfig
from datasynth.prompts import build_synthesis_prompt, parse_generated_samples


@dataclass
class SynthesisResult:
    """Result of data synthesis."""

    success: bool = True
    error: str = ""
    output_path: str = ""
    generated_count: int = 0
    failed_count: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    duration_seconds: float = 0.0


class DataSynthesizer:
    """Generate synthetic data based on schema and seed examples.

    Supports multiple LLM providers (Anthropic, OpenAI) and provides:
    - Batch generation for efficiency
    - Diversity controls
    - Cost estimation and tracking
    - Validation of generated samples
    """

    def __init__(self, config: Optional[SynthesisConfig] = None):
        self.config = config or SynthesisConfig()
        self._client = None
        self._provider = None

    def _init_client(self):
        """Initialize LLM client based on provider."""
        if self._client is not None:
            return

        provider = self.config.provider.lower()

        if provider == "anthropic":
            try:
                import anthropic

                self._client = anthropic.Anthropic()
                self._provider = "anthropic"
            except ImportError:
                raise ImportError("请安装 anthropic: pip install datasynth[anthropic]")

        elif provider == "openai":
            try:
                import openai

                self._client = openai.OpenAI()
                self._provider = "openai"
            except ImportError:
                raise ImportError("请安装 openai: pip install datasynth[openai]")

        else:
            raise ValueError(f"不支持的 provider: {provider}")

    def synthesize(
        self,
        schema: Dict[str, Any],
        seed_samples: List[Dict[str, Any]],
        output_path: str,
        target_count: Optional[int] = None,
        guidelines: Optional[str] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> SynthesisResult:
        """Synthesize data based on schema and seed examples.

        Args:
            schema: Data schema definition
            seed_samples: Seed examples to learn from
            output_path: Output path for generated data
            target_count: Number of samples to generate (overrides config)
            guidelines: Additional generation guidelines
            on_progress: Callback for progress updates (current, total)

        Returns:
            SynthesisResult with generation status
        """
        start_time = datetime.now()
        result = SynthesisResult()

        try:
            self._init_client()

            # Parse schema
            data_schema = DataSchema.from_dict(schema)

            # Determine target count
            count = target_count or self.config.target_count

            # Generate in batches
            all_samples = []
            failed = 0
            total_tokens = 0
            batches = (count + self.config.batch_size - 1) // self.config.batch_size

            for batch_idx in range(batches):
                remaining = count - len(all_samples)
                batch_count = min(self.config.batch_size, remaining)

                if batch_count <= 0:
                    break

                # Select random seed samples for this batch
                selected_seeds = random.sample(
                    seed_samples, min(self.config.seed_sample_count, len(seed_samples))
                )

                # Build prompt
                prompt = build_synthesis_prompt(
                    schema=data_schema,
                    seed_samples=selected_seeds,
                    count=batch_count,
                    guidelines=guidelines,
                    diversity_factor=self.config.diversity_factor,
                )

                # Generate
                try:
                    response_text, tokens = self._call_llm(prompt)
                    total_tokens += tokens

                    # Parse response
                    samples = parse_generated_samples(response_text, data_schema)
                    all_samples.extend(samples)

                except Exception:
                    failed += batch_count
                    # Continue with next batch

                # Progress callback
                if on_progress:
                    on_progress(len(all_samples), count)

            result.generated_count = len(all_samples)
            result.failed_count = failed
            result.total_tokens = total_tokens
            result.estimated_cost = self._estimate_cost(total_tokens)

            # Build output
            output_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "generator": "DataSynth",
                    "version": "0.1.0",
                    "config": {
                        "model": self.config.model,
                        "provider": self.config.provider,
                        "temperature": self.config.temperature,
                        "diversity_factor": self.config.diversity_factor,
                    },
                    "seed_count": len(seed_samples),
                    "generated_count": len(all_samples),
                    "total_tokens": total_tokens,
                    "estimated_cost_usd": result.estimated_cost,
                },
                "schema": schema,
                "samples": [
                    {
                        "id": f"SYNTH_{i + 1:04d}",
                        "data": sample,
                        "synthetic": True,
                    }
                    for i, sample in enumerate(all_samples)
                ],
            }

            # Write output
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            result.output_path = str(output_path)
            result.duration_seconds = (datetime.now() - start_time).total_seconds()

        except Exception as e:
            result.success = False
            result.error = str(e)

        return result

    def synthesize_from_datarecipe(
        self,
        analysis_dir: str,
        output_path: Optional[str] = None,
        target_count: Optional[int] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> SynthesisResult:
        """Synthesize from DataRecipe analysis output.

        Args:
            analysis_dir: Path to DataRecipe analysis output directory
            output_path: Output path (defaults to analysis_dir/11_合成数据/synthetic.json)
            target_count: Number of samples to generate

        Returns:
            SynthesisResult with generation status
        """
        analysis_dir = Path(analysis_dir)

        # Load schema
        schema_path = analysis_dir / "04_复刻指南" / "DATA_SCHEMA.json"
        if not schema_path.exists():
            return SynthesisResult(success=False, error=f"Schema not found: {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        # Load seed samples
        samples_path = analysis_dir / "09_样例数据" / "samples.json"
        if not samples_path.exists():
            return SynthesisResult(success=False, error=f"Seed samples not found: {samples_path}")

        with open(samples_path, "r", encoding="utf-8") as f:
            samples_data = json.load(f)

        seed_samples = []
        for s in samples_data.get("samples", []):
            if "data" in s:
                seed_samples.append(s["data"])
            else:
                seed_samples.append(s)

        if not seed_samples:
            return SynthesisResult(success=False, error="No seed samples found")

        # Load guidelines
        guidelines = None
        guidelines_path = analysis_dir / "03_标注规范" / "ANNOTATION_SPEC.md"
        if guidelines_path.exists():
            guidelines = guidelines_path.read_text(encoding="utf-8")

        # Set output path
        if output_path is None:
            output_dir = analysis_dir / "11_合成数据"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / "synthetic.json"

        return self.synthesize(
            schema=schema,
            seed_samples=seed_samples,
            output_path=str(output_path),
            target_count=target_count,
            guidelines=guidelines,
            on_progress=on_progress,
        )

    def _call_llm(self, prompt: str) -> tuple[str, int]:
        """Call LLM and return response text and token count."""
        if self._provider == "anthropic":
            response = self._client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens_per_sample * self.config.batch_size,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return text, tokens

        elif self._provider == "openai":
            response = self._client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens_per_sample * self.config.batch_size,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content
            tokens = response.usage.total_tokens
            return text, tokens

        else:
            raise ValueError(f"Unknown provider: {self._provider}")

    def _estimate_cost(self, total_tokens: int) -> float:
        """Estimate cost based on token usage."""
        # Rough split: 40% input, 60% output
        input_tokens = total_tokens * 0.4
        output_tokens = total_tokens * 0.6

        input_cost = (input_tokens / 1000) * self.config.input_token_cost
        output_cost = (output_tokens / 1000) * self.config.output_token_cost

        return round(input_cost + output_cost, 4)


class InteractiveSynthesizer:
    """Interactive synthesizer for use with Claude Code / MCP.

    Does not call LLM directly - returns prompts for human/agent to process.
    """

    def __init__(self):
        pass

    def prepare_synthesis(
        self,
        schema: Dict[str, Any],
        seed_samples: List[Dict[str, Any]],
        count: int = 10,
        guidelines: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Prepare synthesis prompt for interactive generation.

        Returns a prompt that can be processed by Claude or another agent.
        """
        data_schema = DataSchema.from_dict(schema)

        prompt = build_synthesis_prompt(
            schema=data_schema,
            seed_samples=seed_samples[:5],  # Limit seed samples
            count=count,
            guidelines=guidelines,
            diversity_factor=0.5,
        )

        return {
            "prompt": prompt,
            "expected_count": count,
            "schema": schema,
            "instructions": (
                "请根据上述 prompt 生成数据。生成完成后，使用 parse_synthesis_result 解析结果。"
            ),
        }

    def parse_result(
        self,
        response_text: str,
        schema: Dict[str, Any],
        output_path: str,
    ) -> SynthesisResult:
        """Parse LLM response and save to file."""
        result = SynthesisResult()

        try:
            data_schema = DataSchema.from_dict(schema)
            samples = parse_generated_samples(response_text, data_schema)

            output_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "generator": "DataSynth (Interactive)",
                    "version": "0.1.0",
                    "generated_count": len(samples),
                },
                "schema": schema,
                "samples": [
                    {
                        "id": f"SYNTH_{i + 1:04d}",
                        "data": sample,
                        "synthetic": True,
                    }
                    for i, sample in enumerate(samples)
                ],
            }

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            result.output_path = str(output_path)
            result.generated_count = len(samples)

        except Exception as e:
            result.success = False
            result.error = str(e)

        return result
