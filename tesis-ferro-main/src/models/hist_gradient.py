from prefect import task
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from skrub import tabular_pipeline


@task(name="entrenar_hist_gradient", log_prints=True)
def train(X_train, y_train):
    print("Configurando HistGradientBoostingClassifier + GridSearchCV...")
    pipe = tabular_pipeline("classifier")

    param_grid = {
        "histgradientboostingclassifier__learning_rate": [0.1, 0.05, 0.02],
        "histgradientboostingclassifier__max_iter": [200, 400, 800],
        "histgradientboostingclassifier__max_leaf_nodes": [31, 63, 127],
        "histgradientboostingclassifier__min_samples_leaf": [10, 20, 40],
        "histgradientboostingclassifier__l2_regularization": [0.0, 0.1, 1.0],
        "histgradientboostingclassifier__early_stopping": [True],
    }

    n_combinaciones = 1
    for v in param_grid.values():
        n_combinaciones *= len(v)
    print(f"Total combinaciones en grid: {n_combinaciones}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, average="macro")

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    print(f"Ejecutando GridSearchCV con 5-fold CV sobre {len(X_train)} muestras...")
    search.fit(X_train, y_train)
    print(f"Mejor score: {search.best_score_:.4f}")
    print(f"Mejores parametros: {search.best_params_}")

    return search
