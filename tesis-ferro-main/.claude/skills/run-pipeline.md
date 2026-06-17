# run-pipeline

Runs the ML and/or LLM default prediction pipeline using Prefect.

## Instructions

The user may specify which pipeline(s) to run and optional parameters. Based on their request:

1. Determine what to run:
   - ML only: `main_flow(run_ml=True, run_llm=False)`
   - LLM only: `main_flow(run_ml=False, run_llm=True, n_samples=200)`
   - Both: `main_flow(run_ml=True, run_llm=True, n_samples=200)`
   - Skip API calls (use cached predictions): add `skip_predict=True`

2. Edit the last line of `main.py` with the desired parameters, then run:
```bash
uv run python main.py
```

3. If Prefect server is not running, remind the user to start it first in a separate terminal:
```bash
uv run prefect server start
```
Prefect UI is at http://127.0.0.1:4200

## Parameters for main_flow

- `run_ml` (bool, default True): Run the 8 ML models
- `run_llm` (bool, default False): Run LLM classification with OpenAI
- `n_samples` (int or None, default 200): Samples for LLM (None = full test set ~2573)
- `model` (str, default "gpt-4o-mini"): OpenAI model to use
- `skip_predict` (bool, default False): Skip LLM API calls, load from cached CSV
- `results_file` (str, default "llm_predictions.csv"): Cache file in llm_tesis/results/

## Notes

- `.env` file must have `OPENAI_API_KEY=sk-...` for LLM pipeline
- ML models saved to `models/` as `.joblib` + `_params.json`
- Plots saved to `plots/`
- All results published as Prefect artifacts in the UI
