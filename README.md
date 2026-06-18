<div align="center">

# 🧠 SABA: Self-Awareness before Action

### Mitigating Logical Inertia via Proactive Cognitive Awareness

[![ACL 2026](https://img.shields.io/badge/ACL-2026_Findings-8A2BE2)](https://2026.aclweb.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.20413-b31b1b.svg)](https://arxiv.org/abs/2604.20413)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-green.svg)](https://www.python.org/)

**Fulong Fan**<sup>1,*</sup> · **Peilin Liu**<sup>1,*</sup> · **Liu FengZhe**<sup>1</sup> · **Shuyan Yang**<sup>2</sup> · **Gang Yan**<sup>2,†</sup>

<sup>1</sup> School of Software, Jilin University &nbsp;|&nbsp; <sup>2</sup> School of Computer Science, Jilin University

<sup>*</sup> Equal contribution &nbsp;|&nbsp; <sup>†</sup> Corresponding author

</div>

---

## ✨ Overview

**SABA** is a novel reasoning framework that introduces a **Self-Awareness Before Action** paradigm for large language models. Unlike traditional "answer-then-correct" approaches (e.g., Self-Refine), SABA proactively audits its own knowledge state *before* committing to a decision — identifying missing premises, logical gaps, and causal inconsistencies before they propagate through the reasoning chain.

In complex, non-interactive narrative reasoning tasks (e.g., detective puzzles), SABA significantly reduces **logical leaps** and suppresses **hallucination** by iteratively constructing a verified knowledge state through structured information fusion and obstacle-driven reasoning.

> 🔥 **Accepted at ACL 2026 Findings.**

---

## 🎯 Key Contributions

- **Self-Awareness Mechanism**: Explicit gap identification before reasoning, preventing premature commitment under incomplete evidence
- **Information Fusion (IF)**: Transforms raw narratives into dense, structured, and verified evidence representations
- **Query-driven Structured Reasoning (QSR)**: Treats missing premises as formal reasoning obstacles, resolved via recursive query decomposition and hypothesis construction
- **Detective Puzzle Benchmark**: A multi-level dataset (Easy / Medium / Complex) for evaluating long-context narrative reasoning under information asymmetry
- **Dual Evaluation Metrics**: Semantic Recall Rate (SRR) and Clue Coverage Rate (CCR) for assessing both answer correctness and evidence grounding

---

## 📊 Performance Highlights

On the **Detective Puzzle Complex split**, SABA (DeepSeek-V3) achieves:

| Metric | SABA | Best Baseline | Gain |
|--------|------|---------------|------|
| Suspect Accuracy (SA) | **79.3%** | 69.8% (GoT) | +9.5% |
| Clue Coverage Rate (CCR) | **83.3%** | 77.1% (S²R) | +6.2% |
| Motive Recall (R-M) | **73.4%** | 69.8% (GoT) | +3.6% |

SABA also achieves **state-of-the-art** on HotpotQA, StrategyQA, and Big-Bench Hard while using **23.3% fewer tokens** than Self-Consistency.

---

## 📁 Repository Structure

```
SABA/
├── SABA.py                     # Main SABA framework implementation
├── dataset/
│   ├── Easy/                   # 5 cases, ~1050 words each
│   ├── Medium/                 # 15 cases, ~1150 words each
│   └── Complex/                # 11 cases, ~950 words each
│       ├── Mystery_text.txt    # Raw case narrative
│       ├── predefined_cues.txt # Gold-standard clue list (for CCR)
│       ├── Answer.txt          # Ground-truth answer
│       └── PostRun_KB/         # SABA output directory
│           ├── q_a.json        # Intermediate reasoning trace
│           └── report.json     # Final prediction results
├── Evaluation_Similarity/      # Semantic recall evaluator
├── requirements.txt            # Python dependencies
└── LICENSE                     # Apache 2.0
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Python 3.12+ recommended
python -m venv saba_env
source saba_env/bin/activate      # Linux / macOS
# Windows: saba_env\Scripts\activate

pip install -r requirements.txt
```

### 2. Configuration

Edit `SABA.py` — locate the configuration section at the top:

```python
# --- Path Configuration ---
PATH_CONFIG = {
    "MODEL_ENCODER_PATH": "path/to/your/encoder",
    "INPUT_PATH": "case/Mystery_text.txt",
    "OUTPUT_QA_PATH": "case/SABA_PostRun_KB/q_a.json",
    "OUTPUT_REPORT_PATH": "case/SABA_PostRun_KB/report.json",
    "GOLD_PATH": "case/Answer.txt",
    "TEST_PATH": "case/report.json",
}

# --- LLM Client Configuration ---
LLM_CONFIG = {
    "api_key": "your-api-key",
    "base_url": "your-base-url",
    "model_name": "deepseek-v3",   # or gemini-1.5-flash, etc.
    "temperature": 0.0,
}
```

### 3. Run

```bash
python SABA.py
```

Two key outputs are generated:
- `q_a.json` — intermediate reasoning trace (for computing Clue Coverage Rate)
- `report.json` — final prediction (for computing Semantic Recall Rate)

---

## 📏 Evaluation

### Metric A — Semantic Recall Rate (SRR)

Evaluates how well the model output matches the gold-standard answer using semantic similarity.

```bash
# Requires: sentence-transformers
python Evaluation_Similarity/Evaluation.py
```

### Metric B — Clue Coverage Rate (CCR)

Measures whether the model identified and utilized the case's critical clues. Check each case directory for:
- `predefined_cues.txt` — the gold-standard clue list
- Manually inspect `q_a.json` for clue coverage against the predefined set

---

## 📖 Dataset

| Difficulty | Cases | Avg. Length | Description |
|------------|-------|-------------|-------------|
| **Easy** | 5 | ~1050 words | Direct inference from explicit clues |
| **Medium** | 15 | ~1150 words | Multi-step causality required |
| **Complex** | 11 | ~950 words | Implicit clues + intentional red herrings |

Each case includes the original narrative, predefined critical clues, and gold-standard answers.

---

## 📜 Citation

If you use the SABA framework or code in your research, please cite:

```bibtex
@misc{fan2026selfawarenessactionmitigatinglogical,
      title={Self-Awareness before Action: Mitigating Logical Inertia via Proactive Cognitive Awareness}, 
      author={Fulong Fan and Peilin Liu and Fengzhe Liu and Shuyan Yang and Gang Yan},
      year={2026},
      eprint={2604.20413},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.20413}, 
}