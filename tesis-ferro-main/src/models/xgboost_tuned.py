import warnings

from prefect import task
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from skrub import TableVectorizer
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


@task(name="entrenar_xgboost_tuned", log_prints=True)
def train(X_train, y_train):
    print("Configurando XGBClassifier + RandomizedSearchCV...")
    model_xgb = XGBClassifier(
        grow_policy="lossguide",
        tree_method="hist",
        enable_categorical=True,
    )

    pipe = Pipeline([
        ("prep", TableVectorizer()),
        ("clf", model_xgb),
    ])

    param_distributions = {
        "clf__max_depth": randint(3, 15),
        "clf__learning_rate": loguniform(0.01, 0.3),
        "clf__n_estimators": randint(1, 500),
        "clf__subsample": uniform(0.5, 0.5),
        "clf__colsample_bytree": uniform(0.5, 0.5),
        "clf__reg_alpha": loguniform(1e-3, 10),
        "clf__reg_lambda": loguniform(1e-3, 10),
        "clf__max_delta_step": randint(0, 10),
        "clf__gamma": uniform(0, 2),
        "clf__min_child_weight": randint(5, 10),
        "clf__scale_pos_weight": uniform(1, 10),
    }

    grid_search_xgb = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=50,
        scoring="f1_weighted",
    )
    print(f"Ejecutando RandomizedSearchCV con 50 iteraciones sobre {len(X_train)} muestras...")
    grid_search_xgb.fit(X_train, y_train)
    print(f"Mejor score: {grid_search_xgb.best_score_:.4f}")
    print(f"Mejores parametros: {grid_search_xgb.best_params_}")

    return grid_search_xgb
