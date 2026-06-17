# show-results

Shows, analyzes, or compares model results from the default prediction project.

## Instructions

When the user asks to see, compare, or summarize model results:

### View results in Prefect UI
Direct the user to http://127.0.0.1:4200 — all artifacts (confusion matrices, ROC curves, classification reports) are published there after each run.

### Run summary comparison
The `src/summary.py` module loads all trained ML models from `models/` and compares them with LLM results:
- Edit `main.py` to run with `run_ml=False, run_llm=False` (summary always runs at the end)
- Or check `src/summary.py` — the `summary_flow` is called at the end of every `main_flow` run

### Check saved artifacts
- **Plots**: `plots/` directory — confusion matrices and ROC curves as PNG
- **Models**: `models/` — `.joblib` files and `_params.json` with best hyperparameters
- **LLM cache**: `llm_tesis/results/llm_predictions.csv` — all LLM predictions with reasoning

### Read LLM predictions
```python
import pandas as pd
df = pd.read_csv("llm_tesis/results/llm_predictions.csv")
print(df[["prediction", "default_probability", "reasoning"]].head())
```

### Load and inspect a trained ML model
```python
import joblib
model = joblib.load("models/xgb_model.joblib")
import json
params = json.load(open("models/xgb_model_params.json"))
```

### Best models
- ML: XGBClassifier and LGBMClassifier (~92-93% accuracy on test set)
- LLM: gpt-4o-mini with Spanish prompts (accuracy depends on n_samples used)
- Test set: ~2,573 samples (2,127 no-default, 446 default)
