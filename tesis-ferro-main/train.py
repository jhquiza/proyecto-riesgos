"""
Reentrenar todos los modelos ML desde cero.

Uso:
    uv run python train.py

- Entrena los modelos definidos en src/ml_pipeline.py (ML_MODELS)
- Guarda los .joblib y _params.json en models/propios/
- Genera el informe comparativo en Prefect al final
"""
from pathlib import Path

from prefect import flow

from src.data import load_data
from src.ml_pipeline import ml_flow
from src.summary import summary_flow

MODELS_DIR = Path(__file__).resolve().parent / "models" / "propios"


@flow(name="Entrenamiento - Prediccion Default", log_prints=True)
def train_flow():
    print(f"Entrenamiento iniciado — destino: {MODELS_DIR}")

    X_train, X_test, y_train, y_test = load_data()
    ml_flow(X_train, X_test, y_train, y_test, MODELS_DIR)
    summary_flow(X_test, y_test, models_dir=MODELS_DIR, model_source="propios")

    print("Entrenamiento completado.")


if __name__ == "__main__":
    train_flow()
