<div align="right">

**English** | [中文](landing-zh.md)

</div>

<div align="center">

<h1>DataSynth</h1>

<h3>LLM-Powered Synthetic Dataset Generation<br/>with Quality-Diversity Optimization</h3>

<p><em>Seed-to-scale synthetic data engine with auto-detected templates, concurrent generation, schema validation, and precise cost estimation</em></p>

<a href="https://github.com/liuxiaotong/data-synth">GitHub</a> ·
<a href="https://pypi.org/project/knowlyr-datasynth/">PyPI</a> ·
<a href="https://knowlyr.com">knowlyr.com</a>

</div>

## Why DataSynth?

High-quality training data is the key bottleneck for LLM performance. Manual annotation is expensive ($0.1--$10 per sample), slow (100 samples/day), and inconsistent across annotators. Naive LLM batch calls lack quality guarantees -- duplicate samples, schema violations, and distribution skew go undetected.

**DataSynth** bridges this gap: starting from ~50 seed samples, it auto-detects data types, selects specialized prompt templates, generates data via concurrent LLM calls, validates against schema constraints, and deduplicates across batches -- all at $0.001--$0.01 per sample.

## Core Features

- **Auto-Detected Templates** -- Automatically identifies instruction-response, preference pairs (DPO/RLHF), or multi-turn dialogue and applies specialized prompts
- **Concurrent Generation** -- Multi-batch parallel LLM calls with thread-safe deduplication and incremental resume (`--resume`)
- **Schema Validation** -- Type checking, range/enum/length constraints; non-compliant samples are filtered automatically
- **Precise Cost Estimation** -- Per-model pricing with `--dry-run` to estimate before generating
- **Post-Generation Hooks** -- Auto-trigger downstream quality checks after generation completes
- **Distribution Statistics** -- Field-level distribution reports for generated datasets

## Quick Start

```bash
pip install knowlyr-datasynth
export ANTHROPIC_API_KEY=your_key

# Generate 100 samples from DataRecipe analysis output
knowlyr-datasynth generate ./analysis_output/my_dataset/ -n 100

# Concurrent generation with cost estimation
knowlyr-datasynth generate ./output/ -n 1000 --concurrency 3 --dry-run

# Resume after interruption
knowlyr-datasynth generate ./output/ -n 1000 --resume

# Interactive mode (no API key needed)
knowlyr-datasynth prepare ./analysis_output/my_dataset/ -n 10
```

```python
from datasynth import SynthEngine

engine = SynthEngine(model="claude-sonnet-4-20250514")
result = engine.generate(
    analysis_dir="./analysis_output/my_dataset/",
    target_count=100,
    concurrency=3,
)
print(f"Generated: {result.generated_count}, Cost: ${result.cost_usd:.4f}")
```

## Pipeline

```mermaid
graph LR
    Seed["Seed Data<br/>(~50 samples)"] --> Detect["Type Detector<br/>Auto-detect"]
    Detect --> Template["Template<br/>Specialized Prompt"]
    Template --> Gen["Generator<br/>Concurrent Batches"]
    Gen --> Val["Validator<br/>Schema Constraints"]
    Val --> Dedup["Deduplicator<br/>Seed + Cross-batch"]
    Dedup --> Stats["Statistics<br/>Distribution Report"]

    style Gen fill:#0969da,color:#fff,stroke:#0969da
    style Val fill:#8b5cf6,color:#fff,stroke:#8b5cf6
    style Dedup fill:#2da44e,color:#fff,stroke:#2da44e
    style Seed fill:#1a1a2e,color:#e0e0e0,stroke:#444
    style Detect fill:#1a1a2e,color:#e0e0e0,stroke:#444
    style Template fill:#1a1a2e,color:#e0e0e0,stroke:#444
    style Stats fill:#1a1a2e,color:#e0e0e0,stroke:#444
```

## Ecosystem

DataSynth is part of the **knowlyr** data infrastructure:

| Layer | Project | Role |
|:---|:---|:---|
| Discovery | **AI Dataset Radar** | Dataset intelligence and trend analysis |
| Analysis | **DataRecipe** | Reverse analysis, schema extraction, cost estimation |
| Production | **DataSynth** | LLM synthesis, auto templates, schema validation, cost estimation |
| Production | **DataLabel** | Zero-server annotation, LLM pre-labeling, IAA analysis |
| Quality | **DataCheck** | Rule validation, anomaly detection, auto-fix |
| Audit | **ModelAudit** | Distillation detection, model fingerprinting |

<div align="center">
<br/>
<a href="https://github.com/liuxiaotong/data-synth">GitHub</a> ·
<a href="https://pypi.org/project/knowlyr-datasynth/">PyPI</a> ·
<a href="https://knowlyr.com">knowlyr.com</a>
<br/><br/>
<sub><a href="https://github.com/liuxiaotong">knowlyr</a> -- LLM-powered synthetic dataset generation with quality-diversity optimization</sub>
</div>
