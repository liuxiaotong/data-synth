"""Core data synthesizer."""

import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from datasynth import __version__
from datasynth.config import DataSchema, SynthesisConfig
from datasynth.prompts import (
    build_synthesis_prompt,
    get_specialized_prompt,
    parse_generated_samples,
)

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Result of data synthesis."""

    success: bool = True
    error: str = ""
    output_path: str = ""
    generated_count: int = 0
    failed_count: int = 0
    dedup_count: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    duration_seconds: float = 0.0
    stats: Dict[str, Any] | None = None


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

    @staticmethod
    def _fingerprint(sample: Dict[str, Any]) -> str:
        """Create a dedup key from a sample by sorting and serializing."""
        return json.dumps(sample, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _compute_stats(
        samples: List[Dict[str, Any]], schema: DataSchema
    ) -> Dict[str, Any]:
        """Compute distribution statistics for generated samples."""
        stats: Dict[str, Any] = {"total_samples": len(samples), "fields": {}}

        field_names = [f.name for f in schema.fields if f.name not in ("id", "metadata")]
        for fname in field_names:
            values = [s.get(fname) for s in samples if fname in s]
            field_stat: Dict[str, Any] = {"count": len(values), "missing": len(samples) - len(values)}

            if not values:
                stats["fields"][fname] = field_stat
                continue

            # String fields: length stats
            if all(isinstance(v, str) for v in values):
                lengths = [len(v) for v in values]
                field_stat["type"] = "text"
                field_stat["avg_length"] = round(sum(lengths) / len(lengths), 1)
                field_stat["min_length"] = min(lengths)
                field_stat["max_length"] = max(lengths)

            # Numeric fields: value stats
            elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                field_stat["type"] = "numeric"
                field_stat["min"] = min(values)
                field_stat["max"] = max(values)
                field_stat["avg"] = round(sum(values) / len(values), 2)
                # Value distribution for integers with small range
                if all(isinstance(v, int) for v in values) and (max(values) - min(values)) <= 20:
                    dist = {}
                    for v in values:
                        dist[str(v)] = dist.get(str(v), 0) + 1
                    field_stat["distribution"] = dict(sorted(dist.items()))

            # List fields
            elif all(isinstance(v, list) for v in values):
                lengths = [len(v) for v in values]
                field_stat["type"] = "list"
                field_stat["avg_items"] = round(sum(lengths) / len(lengths), 1)
                field_stat["min_items"] = min(lengths)
                field_stat["max_items"] = max(lengths)

            else:
                field_stat["type"] = "mixed"

            stats["fields"][fname] = field_stat

        return stats

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

    @staticmethod
    def _load_existing_samples(output_path: str) -> list[Dict[str, Any]]:
        """Load existing samples from an output file (for resume mode)."""
        path = Path(output_path)
        if not path.exists():
            return []

        with open(path, "r", encoding="utf-8") as f:
            if path.suffix == ".jsonl":
                return [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
                if isinstance(data, dict) and "samples" in data:
                    return [s.get("data", s) for s in data["samples"]]
                if isinstance(data, list):
                    return data
        return []

    def synthesize(
        self,
        schema: Dict[str, Any],
        seed_samples: List[Dict[str, Any]],
        output_path: str,
        target_count: Optional[int] = None,
        guidelines: Optional[str] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
        output_format: str = "json",
        resume: bool = False,
    ) -> SynthesisResult:
        """Synthesize data based on schema and seed examples.

        Args:
            schema: Data schema definition
            seed_samples: Seed examples to learn from
            output_path: Output path for generated data
            target_count: Number of samples to generate (overrides config)
            guidelines: Additional generation guidelines
            on_progress: Callback for progress updates (current, total)
            resume: If True, load existing output and continue from where left off

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

            # Build dedup index from seed samples
            seen: set[str] = set()
            if self.config.validate:
                for seed in seed_samples:
                    seen.add(self._fingerprint(seed))
            lock = threading.Lock()

            # Resume: load existing samples
            existing_samples: list[Dict[str, Any]] = []
            if resume:
                existing_samples = self._load_existing_samples(output_path)
                if existing_samples:
                    logger.info("Resume: loaded %d existing samples", len(existing_samples))
                    # Add existing to dedup index
                    if self.config.validate:
                        for s in existing_samples:
                            seen.add(self._fingerprint(s))
                    # Reduce remaining target
                    count = max(0, count - len(existing_samples))
                    if count == 0:
                        result.generated_count = len(existing_samples)
                        result.output_path = str(output_path)
                        result.duration_seconds = (datetime.now() - start_time).total_seconds()
                        return result

            # Shared mutable state
            all_samples: list[Dict[str, Any]] = []
            failed = 0
            deduped = 0
            total_tokens = 0

            # Prepare batch tasks
            num_batches = (count + self.config.batch_size - 1) // self.config.batch_size
            batch_counts = []
            for i in range(num_batches):
                remaining = count - i * self.config.batch_size
                batch_counts.append(min(self.config.batch_size, remaining))

            # Resolve data_type
            data_type = self.config.data_type
            if data_type == "auto":
                data_type = data_schema.detect_data_type()

            def run_batch(batch_idx: int, batch_count: int) -> tuple:
                """Run a single batch. Returns (samples, tokens, failed, deduped)."""
                selected_seeds = random.sample(
                    seed_samples, min(self.config.seed_sample_count, len(seed_samples))
                )

                if data_type:
                    prompt = get_specialized_prompt(
                        data_type=data_type,
                        schema=data_schema,
                        seed_samples=selected_seeds,
                        count=batch_count,
                        guidelines=guidelines,
                    )
                else:
                    prompt = build_synthesis_prompt(
                        schema=data_schema,
                        seed_samples=selected_seeds,
                        count=batch_count,
                        guidelines=guidelines,
                        diversity_factor=self.config.diversity_factor,
                    )

                for attempt in range(1, self.config.max_retries + 1):
                    try:
                        # Increment temperature on retries for more diversity
                        retry_temp = min(
                            self.config.temperature + (attempt - 1) * 0.05, 1.0
                        )
                        temp = retry_temp if attempt > 1 else None
                        response_text, tokens = self._call_llm(prompt, temperature=temp)
                        samples = parse_generated_samples(response_text, data_schema)

                        # Validate + dedup
                        batch_deduped = 0
                        batch_invalid = 0
                        if self.config.validate:
                            # Schema validation
                            valid = []
                            for s in samples:
                                errs = data_schema.validate_sample(s)
                                if errs:
                                    batch_invalid += 1
                                    logger.debug(
                                        "Sample rejected: %s", "; ".join(errs)
                                    )
                                else:
                                    valid.append(s)
                            samples = valid

                            # Thread-safe dedup
                            unique = []
                            with lock:
                                for s in samples:
                                    fp = self._fingerprint(s)
                                    if fp not in seen:
                                        seen.add(fp)
                                        unique.append(s)
                                    else:
                                        batch_deduped += 1
                            samples = unique

                        return samples, tokens, batch_invalid, batch_deduped

                    except Exception as e:
                        logger.warning(
                            "Batch %d attempt %d/%d failed: %s",
                            batch_idx + 1,
                            attempt,
                            self.config.max_retries,
                            e,
                        )
                        if attempt < self.config.max_retries:
                            time.sleep(self.config.retry_delay)

                return [], 0, batch_count, 0

            # Execute batches
            workers = min(self.config.concurrency, num_batches)
            if workers <= 1:
                # Sequential execution
                for batch_idx, bc in enumerate(batch_counts):
                    samples, tokens, batch_failed, batch_deduped = run_batch(batch_idx, bc)
                    all_samples.extend(samples)
                    total_tokens += tokens
                    failed += batch_failed
                    deduped += batch_deduped
                    if on_progress:
                        on_progress(len(all_samples), count)
            else:
                # Concurrent execution
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(run_batch, idx, bc): idx
                        for idx, bc in enumerate(batch_counts)
                    }
                    for future in as_completed(futures):
                        samples, tokens, batch_failed, batch_deduped = future.result()
                        all_samples.extend(samples)
                        total_tokens += tokens
                        failed += batch_failed
                        deduped += batch_deduped
                        if on_progress:
                            on_progress(len(all_samples), count)

            # Merge with existing samples for resume mode
            final_samples = existing_samples + all_samples

            result.generated_count = len(final_samples)
            result.failed_count = failed
            result.dedup_count = deduped
            result.total_tokens = total_tokens
            result.estimated_cost = self._estimate_cost(total_tokens)

            # Build output
            output_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "generator": "DataSynth",
                    "version": __version__,
                    "config": {
                        "model": self.config.model,
                        "provider": self.config.provider,
                        "temperature": self.config.temperature,
                        "diversity_factor": self.config.diversity_factor,
                    },
                    "seed_count": len(seed_samples),
                    "generated_count": len(final_samples),
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
                    for i, sample in enumerate(final_samples)
                ],
            }

            # Write output
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_format == "jsonl":
                with open(output_path, "w", encoding="utf-8") as f:
                    for sample in final_samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            else:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

            result.output_path = str(output_path)
            result.duration_seconds = (datetime.now() - start_time).total_seconds()

            # Compute stats
            if final_samples:
                result.stats = self._compute_stats(final_samples, data_schema)

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
        output_format: str = "json",
        resume: bool = False,
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
            ext = "jsonl" if output_format == "jsonl" else "json"
            output_path = output_dir / f"synthetic.{ext}"

        return self.synthesize(
            schema=schema,
            seed_samples=seed_samples,
            output_path=str(output_path),
            target_count=target_count,
            guidelines=guidelines,
            on_progress=on_progress,
            output_format=output_format,
            resume=resume,
        )

    def _call_llm(self, prompt: str, temperature: float | None = None) -> tuple[str, int]:
        """Call LLM and return response text and token count."""
        temp = temperature if temperature is not None else self.config.temperature

        if self._provider == "anthropic":
            response = self._client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens_per_sample * self.config.batch_size,
                temperature=temp,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return text, tokens

        elif self._provider == "openai":
            response = self._client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens_per_sample * self.config.batch_size,
                temperature=temp,
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

        input_price, output_price = SynthesisConfig._get_pricing(self.config.model)
        input_cost = (input_tokens / 1000) * input_price
        output_cost = (output_tokens / 1000) * output_price

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
        data_type: str = "auto",
    ) -> Dict[str, Any]:
        """Prepare synthesis prompt for interactive generation.

        Returns a prompt that can be processed by Claude or another agent.
        """
        data_schema = DataSchema.from_dict(schema)

        # Resolve data type
        resolved_type = data_type
        if resolved_type == "auto":
            resolved_type = data_schema.detect_data_type()

        if resolved_type:
            prompt = get_specialized_prompt(
                data_type=resolved_type,
                schema=data_schema,
                seed_samples=seed_samples[:5],
                count=count,
                guidelines=guidelines,
            )
        else:
            prompt = build_synthesis_prompt(
                schema=data_schema,
                seed_samples=seed_samples[:5],
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
                    "version": __version__,
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
