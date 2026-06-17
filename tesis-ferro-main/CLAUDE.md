# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's thesis project for **default prediction** using WhatsApp Business data. Binary classification task predicting customer default on an imbalanced dataset. Two pipelines: traditional ML models and LLM-based prediction (OpenAI).

## Running the Project

### Via uv (recommended)

```bash
# Start Prefect server (separate terminal)
uv run prefect server start

# Run both pipelines (edit main.py last line)
uv run python main.py
```

**main_flow parameters:**
```python
main_flow(
    run_ml=True,           # Run 8 ML models
    run_llm=False,         # Run LLM classification
    n_samples=200,         # LLM samples (None = full test set ~2573)
    model="gpt-4o-mini",   # OpenAI model
    skip_predict=False,    # True = use cached CSV, skip API calls
    results_file="llm_predictions.csv"
)
```

### Common run modes

```python
# ML only (default)
main_flow(run_ml=True, run_llm=False)

# LLM only (uses cache)
main_flow(run_ml=False, run_llm=True, skip_predict=True)

# LLM only (calls API)
main_flow(run_ml=False, run_llm=True, n_samples=200)

# Both pipelines
main_flow(run_ml=True, run_llm=True, n_samples=None)
```

### Via Jupyter notebook (legacy)

- Open and run `base/Modelos de Entrenamiento WhatsApp Business.ipynb` in Jupyter

### Data file

`base/WhatsApp Business Data (1).parquet`

### Environment variables

`.env` file at project root with `OPENAI_API_KEY=sk-...` (loaded via python-dotenv)

### Key Python Dependencies

scikit-learn, skrub (TableVectorizer), lightgbm, xgboost, pandas, joblib, imbalanced-learn (imblearn), scipy, matplotlib, numpy, prefect, openai, tqdm, python-dotenv

Managed via `uv` — see `pyproject.toml`.

## Architecture

### Orchestration

All functions are decorated with Prefect `@task(log_prints=True)` and entry points with `@flow(log_prints=True)`. Plots and reports are published as Prefect artifacts (markdown with base64 images, table artifacts). Prefect UI available at `http://127.0.0.1:4200`.

### ML Pipeline Pattern

All models use sklearn `Pipeline` with two stages:
1. **Preprocessing**: `skrub.TableVectorizer()` for automated categorical/numeric feature handling
2. **Classifier**: The model being trained

### Models Compared (8 ML models)

| Artifact name     | Model                            | Tuning              |
|-------------------|----------------------------------|---------------------|
| lgbm_skrub        | LGBMClassifier                   | Pre-optimized params |
| lgb_model         | LGBMClassifier                   | RandomizedSearchCV  |
| hist_model        | HistGradientBoostingClassifier   | GridSearchCV        |
| xgb_model         | XGBClassifier                    | RandomizedSearchCV  |
| rfc_model         | RandomForestClassifier           | RandomizedSearchCV  |
| lrc_model         | LogisticRegression               | RandomizedSearchCV  |
| lrc_bal_model     | LogisticRegression + UnderSampler| RandomizedSearchCV  |
| mlp_model         | MLPClassifier                    | RandomizedSearchCV  |

Best performers: XGBClassifier and LGBMClassifier (~92-93% accuracy).

### LLM Pipeline

Row-by-row classification using OpenAI API (`gpt-4o-mini` default). Each row is sent with a Spanish-language system prompt describing the credit risk task and feature meanings. Responses include `prediction`, `default_probability`, and `reasoning`. Results cached to `llm_tesis/results/llm_predictions.csv` to avoid re-running API calls.

### Summary & Comparison

`src/summary.py` — `summary_flow` runs at the end of every `main_flow`. Loads all trained ML models from `models/`, loads LLM results from CSV cache, generates comparative ROC curves, and publishes ranking tables as Prefect artifacts.

### Hyperparameter Tuning

- `RandomizedSearchCV` and `GridSearchCV` with `StratifiedKFold(n_splits=5)`
- Scoring metric: `f1_weighted`

### Class Imbalance Handling

- `class_weight='balanced'` for sklearn models
- `scale_pos_weight` for gradient boosting models
- `RandomUnderSampler` from imblearn for logistic regression balanced variant

### Model Artifacts

Trained models saved as `.joblib` files with corresponding `_params.json` files in the `models/` directory (e.g., `xgb_model.joblib` + `xgb_model_params.json`).

### Data Split

80/20 train/test split with stratification on the `default` target column. Test set: ~2,573 samples (2,127 no-default, 446 default).

## Project Structure

```
main.py                  # Top-level Prefect flow — orchestrates ML and LLM pipelines
src/
├── __init__.py
├── data.py              # Data loading and train/test split (@task)
├── evaluation.py        # Classification report + confusion matrix + Prefect artifacts (@task)
├── persistence.py       # Save model (joblib) + params (JSON) (@task)
├── summary.py           # Comparative summary: all ML models + LLM (@flow)
└── models/
    ├── __init__.py
    ├── lgbm_skrub.py              # LGBMClassifier with pre-optimized params (@task)
    ├── lgbm_tuned.py              # LGBMClassifier with RandomizedSearchCV (@task)
    ├── hist_gradient.py           # HistGradientBoosting with GridSearchCV (@task)
    ├── xgboost_tuned.py           # XGBClassifier with RandomizedSearchCV (@task)
    ├── random_forest.py           # RandomForestClassifier with RandomizedSearchCV (@task)
    ├── logistic_regression.py     # LogisticRegression with RandomizedSearchCV (@task)
    ├── logistic_regression_balanced.py  # LogisticRegression + RandomUnderSampler (@task)
    └── mlp.py                     # MLPClassifier with RandomizedSearchCV (@task)
llm_tesis/
├── __init__.py
├── prompt.py            # System/user prompt templates with Spanish feature descriptions
├── client.py            # OpenAI API wrapper with retry/backoff (max 5 retries) (@task)
├── predict.py           # Batch prediction, stratified sampling, CSV caching (@task)
├── evaluate.py          # ROC curves, classification report, confusion matrix + Prefect artifacts (@task)
├── main.py              # LLM pipeline Prefect flow (@flow)
└── results/
    └── llm_predictions.csv   # Cached LLM predictions (all features + reasoning)
plots/                   # Generated plots (confusion matrices, ROC curves)
models/                  # Saved ML model artifacts (.joblib + _params.json)
base/                    # Legacy notebooks + parquet data file
.claude/
├── CLAUDE.md            # This file
└── skills/
    ├── run-pipeline.md  # /run-pipeline skill: run ML/LLM pipelines
    ├── add-model.md     # /add-model skill: add a new ML model
    ├── show-results.md  # /show-results skill: view/compare results
    └── build-thesis.md  # /build-thesis skill: regenerate plots + compile PDF
```

## Conventions

- Project language is **Spanish** (comments, variable names, print statements)
- Original notebook kept in `base/` for reference; modularized code in `src/`
- All functions use `@task(log_prints=True)`, all entry points use `@flow(log_prints=True)`
- No argparse — pipeline parameters are set directly in `main.py` last line
- `random_state=1` used throughout for reproducibility

## Available Skills (slash commands)

- `/run-pipeline` — Run ML and/or LLM pipelines with correct parameters
- `/add-model` — Add a new ML classifier following the project pattern
- `/show-results` — View, compare, or analyze model results
- `/build-thesis` — Regenerate plots (English labels), update LaTeX tables, compile PDF with tectonic
