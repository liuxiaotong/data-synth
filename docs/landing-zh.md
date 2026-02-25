<div align="right">

[English](landing-en.md) | **中文**

</div>

<div align="center">

<h1>DataSynth</h1>

<h3>LLM 驱动的合成数据生成引擎<br/>质量-多样性优化</h3>

<p><em>种子驱动的合成数据引擎——自动模板检测、并发生成、Schema 验证、精确成本估算</em></p>

<a href="https://github.com/liuxiaotong/data-synth">GitHub</a> ·
<a href="https://pypi.org/project/knowlyr-datasynth/">PyPI</a> ·
<a href="https://knowlyr.com">knowlyr.com</a>

</div>

## 为什么选择 DataSynth？

高质量训练数据是 LLM 性能的关键瓶颈。人工标注成本高（$0.1--$10/条）、速度慢（100 条/天）、一致性差（标注员理解差异）。而简单的 LLM 批量调用又缺少质量保证——重复样本、违反 Schema 约束、分布偏斜等问题无法自动检测。

**DataSynth** 解决这一问题：从约 50 条种子数据出发，自动检测数据类型、选用专用 Prompt 模板、并发调用 LLM 生成、Schema 约束验证、跨批次去重——每条成本仅 $0.001--$0.01。

## 核心特性

- **自动模板检测** —— 自动识别指令-回复、偏好对（DPO/RLHF）或多轮对话，匹配专用 Prompt
- **并发生成** —— 多批次并行 LLM 调用，线程安全去重，支持增量续跑（`--resume`）
- **Schema 验证** —— 类型检查、范围/枚举/长度约束，不合规样本自动过滤
- **精确成本估算** —— 按模型定价计算，`--dry-run` 先估再生
- **后置钩子** —— 生成完成后自动触发下游质检
- **分布统计** —— 字段级分布报告

## 快速开始

```bash
pip install knowlyr-datasynth
export ANTHROPIC_API_KEY=your_key

# 从 DataRecipe 分析结果生成 100 条数据
knowlyr-datasynth generate ./analysis_output/my_dataset/ -n 100

# 并发生成 + 成本估算
knowlyr-datasynth generate ./output/ -n 1000 --concurrency 3 --dry-run

# 中断后续跑
knowlyr-datasynth generate ./output/ -n 1000 --resume

# 交互模式（无需 API key）
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
print(f"已生成: {result.generated_count}, 成本: ${result.cost_usd:.4f}")
```

## 管线

```mermaid
graph LR
    Seed["种子数据<br/>(~50 条样本)"] --> Detect["类型检测器<br/>自动检测"]
    Detect --> Template["模板<br/>专用 Prompt"]
    Template --> Gen["生成器<br/>并发批量"]
    Gen --> Val["验证器<br/>Schema 约束"]
    Val --> Dedup["去重器<br/>种子集 + 跨批次"]
    Dedup --> Stats["统计<br/>分布报告"]

    style Gen fill:#0969da,color:#fff,stroke:#0969da
    style Val fill:#8b5cf6,color:#fff,stroke:#8b5cf6
    style Dedup fill:#2da44e,color:#fff,stroke:#2da44e
    style Seed fill:#1a1a2e,color:#e0e0e0,stroke:#444
    style Detect fill:#1a1a2e,color:#e0e0e0,stroke:#444
    style Template fill:#1a1a2e,color:#e0e0e0,stroke:#444
    style Stats fill:#1a1a2e,color:#e0e0e0,stroke:#444
```

## 生态系统

DataSynth 是 **knowlyr** 数据基础设施的一部分：

| 层 | 项目 | 职责 |
|:---|:---|:---|
| 发现 | **AI Dataset Radar** | 数据集竞争情报、趋势分析 |
| 分析 | **DataRecipe** | 逆向分析、Schema 提取、成本估算 |
| 生产 | **DataSynth** | LLM 合成 · 智能模板 · Schema 验证 · 成本精算 |
| 生产 | **DataLabel** | 零服务器标注 · LLM 预标注 · IAA 分析 |
| 质量 | **DataCheck** | 规则验证、异常检测、自动修复 |
| 审计 | **ModelAudit** | 蒸馏检测、模型指纹 |

<div align="center">
<br/>
<a href="https://github.com/liuxiaotong/data-synth">GitHub</a> ·
<a href="https://pypi.org/project/knowlyr-datasynth/">PyPI</a> ·
<a href="https://knowlyr.com">knowlyr.com</a>
<br/><br/>
<sub><a href="https://github.com/liuxiaotong">knowlyr</a> — LLM 驱动的合成数据生成引擎，质量-多样性优化</sub>
</div>
