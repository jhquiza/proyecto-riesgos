from lightgbm import LGBMClassifier
from prefect import task
from skrub import tabular_pipeline


@task(name="entrenar_lgbm_skrub", log_prints=True)
def train(X_train, y_train):
    print("Configurando LGBMClassifier con parametros pre-optimizados + tabular_pipeline...")
    lgb_model_best = LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        colsample_bytree=0.9005463006833063,
        learning_rate=0.12023779811254239,
        max_depth=10,
        min_child_samples=39,
        n_estimators=105,
        num_leaves=64,
        reg_alpha=0.096,
        reg_lambda=0.003019661310620262,
        scale_pos_weight=2.007887722293911,
        subsample=0.7707980192130739,
        verbose=0,
    )

    model_skrub = tabular_pipeline(lgb_model_best)
    print(f"Entrenando con {len(X_train)} muestras...")
    model_skrub.fit(X_train, y_train)
    print("Entrenamiento completado")

    return model_skrub
