# gtp-hipotesis

Knowledge base for the paper proposing the LLM hypothesis this thesis tests.

---

## Reference

**Babaei, G., & Giudici, P.**
*GPT classifications, with application to credit lending.*
Machine Learning with Applications, 16, 100534 (2024).
https://doi.org/10.1016/j.mlwa.2024.100534
University of Pavia, Italy.

---

## Core Claim

> An **Informed GPT** model (few-shot, 80 in-context training examples) approaches the AUC of a classical Logistic Regression model trained on **30,000 observations** — a 375x economy of data.

---

## Three Approaches Compared

| Approach | Training data used | Mean AUC | Min AUC | Max AUC | Std |
|---|---|---|---|---|---|
| **Informed GPT** | 80 labeled examples in prompt | 0.667 | 0.626 | 0.696 | 0.026 |
| GPT (zero-shot) | None (pre-training only) | 0.613 | 0.590 | 0.645 | 0.021 |
| Logistic Regression | 30,000 observations | 0.753 | 0.702 | 0.795 | 0.031 |

Each model repeated **5 times** with different random samples. Same ~170 test observations per iteration.

---

## Dataset

| Property | Value |
|---|---|
| Source | LendingClub platform (Kaggle) |
| Year | 2018 |
| Task | Binary: loan Accepted / Rejected |
| Imbalance ratio | 0.00185 (Accepted / Rejected) — extremely imbalanced |
| Features used | Loan Amount, Loan Title (DTI), Employment Length |
| Preprocessing | Random undersampling to balance classes; missing rows removed |
| Variable encoding | Employment Length → Junior / Experienced; Title → Debt / Personal |
| Train size | 30,000 (for LR); 0–80 (for GPT) |
| Test size | ~170 per iteration |

---

## Prompt Structure (Fig. 3)

Four sections, in order:

```
[1] Problem definition
    Define the binary classification task and the two output classes
    (Accepted / Rejected)

[2] Examples  ← ONLY in Informed GPT, removed in zero-shot
    Stratified subsample from training data
    Format: feature values → label

[3] Variable descriptions
    Define each feature and its possible values

[4] Request
    The test observation to classify → ask for output label
```

One prompt per test observation → one API call per observation.

---

## Effect of Example Size (Table 2 — Informed GPT only, 5 runs each)

| Example size | Mean AUC | Min AUC | Max AUC | Std |
|---|---|---|---|---|
| 40 | 0.627 | 0.604 | 0.665 | 0.021 |
| 48 | 0.607 | 0.577 | 0.653 | 0.027 |
| 56 | 0.593 | 0.564 | 0.641 | 0.030 |
| 64 | 0.609 | 0.577 | 0.639 | 0.026 |
| 72 | 0.621 | 0.553 | 0.692 | 0.044 |
| **80** | **0.667** | **0.626** | **0.696** | **0.026** |

- AUC does **not** increase monotonically (sampling variability with small N)
- Best result at 80 examples
- Authors project that ~640 examples could surpass LR trained on 30,000

---

## Experimental Design

- **Repetitions**: 5 runs per model, with different random train/test samples
- **Test size**: ~170 observations per run (same for all 3 models within each run)
- **Train size**: 30,000 (LR), 80 (Informed GPT), 0 (GPT)
- **Metric**: AUC — reports Min, Max, Mean, Std across 5 runs
- **API**: OpenAI API, one request per test observation

---

## Key Findings

1. **Zero-shot GPT** (no examples) achieves AUC ~0.61 — meaningful but below LR
2. **Informed GPT** (80 examples) reaches AUC ~0.67 — significantly closer to LR
3. **LR** achieves AUC ~0.75 using 30,000 training observations
4. Adding just 80 in-context examples closes ~54% of the gap between zero-shot GPT and LR
5. Results are not monotonically increasing with example size due to sampling variability at small N
6. Informed GPT uses **375x fewer training observations** than LR for comparable (not equal) performance

---

## Conclusions & Limitations

- LLMs are **not recommended as drop-in replacements** for LR in simple classification — they are more complex and costly
- The paper's point: even when GPT is "not suited," it achieves reasonable performance with far less data
- Practical advantage: **data economy and privacy** — fewer labeled examples needed
- GPT-based approaches are accessible to non-data-scientists
- High computational cost of API calls limits test set size
- Robustness not fully characterized (high Std relative to AUC differences)

---

## Relation to This Thesis

| Paper | This Thesis |
|---|---|
| LendingClub loan acceptance/rejection | WhatsApp Business → credit default prediction |
| 4 features | ~26 features (richer sociodemographic + behavioral data) |
| Zero-shot GPT vs Informed GPT vs LR | Both implemented — see `llm_tesis/experiment.py` |
| Random undersampling | class_weight + RandomUnderSampler |
| Evaluation: AUC only | Evaluation: AUC + F1 + confusion matrix + Prefect artifacts |
| Models: GPT + LR | Models: GPT + LR + LightGBM + XGBoost + RF + MLP + HistGBM |
| Dataset: lending platform | Dataset: cooperative sector (aligned with seminal paper) |

**Hypothesis to test**: Does Informed GPT (few-shot) also outperform zero-shot GPT and approach ML model performance in the WhatsApp Business default prediction context?

## Implementation in Codebase

### New files / functions
- `llm_tesis/experiment.py` → `run_experiment()` flow — main experiment replication
- `llm_tesis/prompt.py` → `build_few_shot_section(examples_X, examples_y)` — formats training examples for the prompt
- `llm_tesis/predict.py` → `sample_few_shot_examples()` task + updated `predict_batch(few_shot_X, few_shot_y)`
- `llm_tesis/client.py` → updated `classify_row(few_shot_text)` — injects examples into system prompt

### How to run
```python
# main.py last line:
main_flow(
    run_ml=False,
    run_llm=False,
    run_experiment=True,
    experiment_n_test=170,      # test obs per run (paper: 170)
    experiment_n_runs=5,        # repetitions (paper: 5)
    experiment_example_sizes=[0, 40, 48, 56, 64, 72, 80],  # 0 = zero-shot
)
```

Or standalone:
```bash
uv run python -m llm_tesis.experiment
```

### Outputs (Prefect artifacts)
- `experimento-informed-gpt-detalle` — Table 2 from paper (AUC by example size)
- `experimento-informed-gpt-resumen` — Table 1 from paper (GPT vs Informed GPT vs LR)
- `experimento-informed-gpt-analisis` — Markdown with key findings
- `llm_tesis/results/experiment_results.csv` — raw results (resumable if interrupted)