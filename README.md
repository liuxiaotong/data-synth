<div align="center">

# DataSynth

**数据合成工具 - 基于种子数据批量生成高质量训练数据**

[![PyPI](https://img.shields.io/pypi/v/datasynth?color=blue)](https://pypi.org/project/datasynth/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-4_Tools-purple.svg)](#mcp-server)

[快速开始](#快速开始) · [交互模式](#交互模式) · [MCP Server](#mcp-server) · [与 DataRecipe 联动](#与-datarecipe-联动)

</div>

---

基于少量种子数据和 Schema 定义，使用 LLM 批量生成高质量训练数据。支持 API 模式和交互模式。

## 核心能力

```
Schema + 种子数据 (50条) → LLM 合成 → 批量数据 (1000+条) → 质检筛选
```

### 解决的问题

| 痛点 | 传统方案 | DataSynth |
|------|----------|-----------|
| **成本** | 人工标注 $0.1-$10/条 | LLM 生成 $0.001-$0.01/条 |
| **速度** | 人工 100条/天 | 自动 10000条/小时 |
| **规模** | 需要招人、培训 | 按需弹性生成 |
| **一致性** | 标注员理解差异 | 规则 + 模板保证一致 |

### 工作模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **API 模式** | 直接调用 LLM API | 有 API key，批量生成 |
| **交互模式** | 生成 Prompt，手动调用 | Claude Code 中使用，无需 API key |

## 安装

```bash
pip install datasynth
```

可选依赖：

```bash
pip install datasynth[anthropic]  # Anthropic Claude
pip install datasynth[openai]     # OpenAI GPT
pip install datasynth[llm]        # 两者都装
pip install datasynth[mcp]        # MCP 服务器
pip install datasynth[all]        # 全部功能
```

## 快速开始

### API 模式 (需要 API key)

```bash
# 设置 API key
export ANTHROPIC_API_KEY=your_key

# 从 DataRecipe 分析结果生成
datasynth generate ./analysis_output/my_dataset/ -n 100

# 估算成本
datasynth generate ./analysis_output/my_dataset/ -n 1000 --dry-run
```

<details>
<summary>输出示例</summary>

```
正在从 ./analysis_output/my_dataset/ 生成合成数据...
  目标数量: 100
  模型: claude-sonnet-4-20250514
  进度: 100/100
✓ 生成成功: ./analysis_output/my_dataset/11_合成数据/synthetic.json
  生成数量: 100
  失败数量: 0
  Token 用量: 45,230
  预计成本: $0.1823
  耗时: 42.3s
```

</details>

### 交互模式 (无需 API key)

```bash
# 生成 Prompt
datasynth prepare ./analysis_output/my_dataset/ -n 10

# 将 Prompt 复制到 Claude，获取结果后解析
```

在 Claude Code 中使用更方便，见 [MCP Server](#mcp-server) 章节。

---

## 成本估算

```bash
datasynth estimate -n 1000
```

```
成本估算:
  目标数量: 1000
  预计批次: 200
  预计输入 Token: 400,000
  预计输出 Token: 600,000
  预计成本: $10.20
  模型: claude-sonnet-4-20250514
```

### 不同规模的成本参考

| 数量 | 预计成本 | 预计时间 |
|------|----------|----------|
| 100 | ~$1 | ~1 分钟 |
| 1,000 | ~$10 | ~10 分钟 |
| 10,000 | ~$100 | ~2 小时 |

---

## 交互模式

交互模式适合在 Claude Code 中使用，不需要 API key：

### 步骤 1: 准备 Prompt

```bash
datasynth prepare ./analysis_output/my_dataset/ -n 10
```

### 步骤 2: 将 Prompt 发送给 Claude

复制输出的 Prompt，发送给 Claude 生成数据。

### 步骤 3: 解析结果

使用 MCP 工具 `parse_synthesis_result` 解析 Claude 的回复。

---

## MCP Server

在 Claude Desktop / Claude Code 中直接使用。

### 配置

添加到 `~/Library/Application Support/Claude/claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "datasynth": {
      "command": "uv",
      "args": ["--directory", "/path/to/data-synth", "run", "python", "-m", "datasynth.mcp_server"]
    }
  }
}
```

### 可用工具

| 工具 | 功能 |
|------|------|
| `prepare_synthesis` | 准备合成 Prompt（交互模式） |
| `parse_synthesis_result` | 解析 LLM 生成结果并保存 |
| `synthesize_data` | 直接调用 LLM 生成（需要 API key） |
| `estimate_synthesis_cost` | 估算生成成本 |

### 使用示例 (交互模式)

```
用户: 帮我基于 ./output/SVGEditBench 生成 20 条合成数据

Claude: [调用 prepare_synthesis]
        生成 Prompt...

        [Claude 自己执行 Prompt 生成数据]

        [调用 parse_synthesis_result]
        ✓ 合成数据已保存:
        - 输出路径: ./output/SVGEditBench/11_合成数据/synthetic.json
        - 生成数量: 20
```

---

## 与 DataRecipe 联动

DataSynth 是 DataRecipe 生态的一部分：

```
DataRecipe (分析) → DataLabel (标注) → DataSynth (合成) → DataCheck (质检)
     ↓                    ↓                  ↓
  提取 Schema        标注种子数据        批量生成
  生成样例 50条      质量校准           扩充到 1000+
```

### 完整工作流

```bash
# 1. DataRecipe: 分析数据集
datarecipe deep-analyze tencent/CL-bench -o ./output

# 2. DataLabel: 生成标注界面，人工标注种子数据
datalabel generate ./output/tencent_CL-bench/

# 3. DataSynth: 批量合成
datasynth generate ./output/tencent_CL-bench/ -n 1000

# 4. DataCheck: 质量检查 (TODO)
```

### 三 MCP 配置

```json
{
  "mcpServers": {
    "datarecipe": {
      "command": "uv",
      "args": ["--directory", "/path/to/data-recipe", "run", "datarecipe-mcp"]
    },
    "datalabel": {
      "command": "uv",
      "args": ["--directory", "/path/to/data-label", "run", "python", "-m", "datalabel.mcp_server"]
    },
    "datasynth": {
      "command": "uv",
      "args": ["--directory", "/path/to/data-synth", "run", "python", "-m", "datasynth.mcp_server"]
    }
  }
}
```

---

## 命令参考

| 命令 | 功能 |
|------|------|
| `datasynth generate <dir>` | 从 DataRecipe 分析结果生成 (API 模式) |
| `datasynth generate <dir> --dry-run` | 仅估算成本 |
| `datasynth create <schema> <seeds> -o <out>` | 从自定义文件生成 |
| `datasynth prepare <dir>` | 准备 Prompt (交互模式) |
| `datasynth estimate -n <count>` | 估算成本 |

### 生成选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-n, --count` | 生成数量 | 100 |
| `-m, --model` | LLM 模型 | claude-sonnet-4-20250514 |
| `-p, --provider` | 提供商 | anthropic |
| `-t, --temperature` | 采样温度 | 0.8 |
| `--batch-size` | 每批数量 | 5 |

---

## API 使用

```python
from datasynth import DataSynthesizer, SynthesisConfig

# 配置
config = SynthesisConfig(
    target_count=100,
    model="claude-sonnet-4-20250514",
    provider="anthropic",
    temperature=0.8,
)

# 生成
synthesizer = DataSynthesizer(config)
result = synthesizer.synthesize_from_datarecipe(
    analysis_dir="./output/my_dataset/",
)

print(f"生成数量: {result.generated_count}")
print(f"成本: ${result.estimated_cost:.4f}")
```

---

## 项目架构

```
src/datasynth/
├── synthesizer.py    # 核心合成器
├── prompts.py        # Prompt 模板和解析
├── config.py         # 配置和 Schema
├── cli.py            # CLI 命令行
└── mcp_server.py     # MCP Server (4 工具)
```

---

## License

[MIT](LICENSE)

---

<div align="center">
<sub>为数据团队提供低成本、高效率的数据扩充方案</sub>
</div>
