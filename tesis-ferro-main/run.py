"""
Correr el pipeline usando modelos ya entrenados (joblibs).

Uso:
    uv run python run.py

Tres tecnicas comparables:
    ML          — modelos sklearn/XGBoost/LightGBM precargados (propios o jhquiza)
    Zero-shot   — LLM clasifica sin ejemplos en el prompt
    Informed GPT — LLM con N ejemplos reales del train en el prompt (Babaei & Giudici, 2024)

Parametros configurables al final del archivo.
"""
from pathlib import Path

from prefect import flow

from src.data import load_data
from src.summary import summary_flow
from llm_tesis.zero_shot.flow import zero_shot_flow
from llm_tesis.informed_gpt.flow import informed_gpt_flow

_BASE_DIR = Path(__file__).resolve().parent

MODEL_DIRS = {
    "propios": _BASE_DIR / "models" / "propios",
    "jhquiza": _BASE_DIR / "models" / "jhquiza",
}

@flow(name="Corrida — Prediccion de Default", log_prints=True)
def run_flow(
    model_source="jhquiza",
    # Zero-shot LLM
    run_zero_shot=False,
    zero_shot_n_samples=200,
    zero_shot_skip_predict=False,
    zero_shot_results_file="llm_predictions.csv",
    # Informed GPT (Babaei & Giudici, 2024)
    run_informed_gpt=True,
    informed_gpt_n_runs=5,
    informed_gpt_example_sizes=None,
    informed_gpt_results_file="experiment_results.csv",
    # Modelo OpenAI (compartido entre zero-shot e informed GPT)
    model="gpt-4o-mini",
):
    if model_source not in MODEL_DIRS:
        raise ValueError(f"model_source debe ser: {list(MODEL_DIRS.keys())}")

    models_dir = MODEL_DIRS[model_source]

    print(f"Corrida iniciada | modelos={model_source} | zero-shot={run_zero_shot} | informed-gpt={run_informed_gpt}")

    X_train, X_test, y_train, y_test = load_data()

    if run_zero_shot:
        zero_shot_flow(
            X_test, y_test,
            n_samples=zero_shot_n_samples,
            model=model,
            skip_predict=zero_shot_skip_predict,
            results_file=zero_shot_results_file,
        )

    if run_informed_gpt:
        informed_gpt_flow(
            X_train, X_test, y_train, y_test,
            model=model,
            n_runs=informed_gpt_n_runs,
            example_sizes=informed_gpt_example_sizes,
            results_file=informed_gpt_results_file,
            models_dir=models_dir,
        )

    summary_flow(
        X_test, y_test,
        models_dir=models_dir,
        model_source=model_source,
        llm_results_file=zero_shot_results_file,
    )

    print("Corrida completada.")


if __name__ == "__main__":
    run_flow(
        model_source="propios",
        run_informed_gpt=True,
        informed_gpt_n_runs=5,
        informed_gpt_example_sizes=[10, 20, 40, 80],
    )
