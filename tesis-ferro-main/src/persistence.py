import json
from pathlib import Path

import joblib
from prefect import task


@task(name="guardar_modelo", log_prints=True)
def save_model(model, name, models_dir, params=None):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{name}.joblib"
    joblib.dump(model, model_path)
    print(f"Modelo guardado: {model_path}")

    if params is not None:
        params_path = models_dir / f"{name}_params.json"
        with open(params_path, "w") as f:
            json.dump(params, f, indent=4)
        print(f"Parametros guardados: {params_path}")
        print(f"Mejores parametros: {params}")
