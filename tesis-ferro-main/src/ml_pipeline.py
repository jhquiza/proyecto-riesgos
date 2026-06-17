from pathlib import Path

from prefect import flow, task

from src.data import load_data
from src.evaluation import evaluate
from src.models import (
    hist_gradient,
    lgbm_skrub,
    lgbm_tuned,
    logistic_regression,
    logistic_regression_balanced,
    mlp,
    random_forest,
    xgboost_tuned,
)
from src.persistence import save_model

ML_MODELS = [
    ("lgbm_skrub", lgbm_skrub, None),
    ("lgb_model", lgbm_tuned, "best_estimator_"),
    ("hist_model", hist_gradient, "best_estimator_"),
    ("xgb_model", xgboost_tuned, "best_estimator_"),
    ("rfc_model", random_forest, "best_estimator_"),
    #("lrc_model", logistic_regression, "best_estimator_"),
    #("lrc_bal_model", logistic_regression_balanced, "best_estimator_"),
    #("mlp_model", mlp, "best_estimator_"),
]


@task(name="entrenar_y_evaluar_modelo", log_prints=True)
def train_and_evaluate(name, module, best_attr, X_train, X_test, y_train, y_test, models_dir):
    print(f"Modelo: {name}")

    result = module.train(X_train, y_train)

    if best_attr is not None:
        model = getattr(result, best_attr)
        params = result.best_params_
        print(f"Mejor score CV: {result.best_score_:.4f}")
    else:
        model = result
        params = None

    evaluate(model, X_test, y_test, model_name=name)
    save_model(model, name, models_dir, params=params)
    print(f"Modelo {name} completado")


@flow(name="Pipeline ML", log_prints=True)
def ml_flow(X_train, X_test, y_train, y_test, models_dir):
    print(f"Pipeline ML: {len(ML_MODELS)} modelos | train={len(X_train)} test={len(X_test)} | destino={models_dir}")

    for name, module, best_attr in ML_MODELS:
        train_and_evaluate(name, module, best_attr, X_train, X_test, y_train, y_test, models_dir)

    print("Pipeline ML completado.")
