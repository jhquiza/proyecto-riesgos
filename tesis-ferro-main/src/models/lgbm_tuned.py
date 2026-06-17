import warnings

from lightgbm import LGBMClassifier
from prefect import task
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from skrub import TableVectorizer

warnings.filterwarnings("ignore")


@task(name="entrenar_lgbm_tuned", log_prints=True)
def train(X_train, y_train):
    print("Configurando LGBMClassifier + RandomizedSearchCV...")
    pipe = Pipeline([
        ("prep", TableVectorizer()),
        ("clf", LGBMClassifier(objective="binary", random_state=42)),
    ])

    param_distributions = {
        "clf__num_leaves": randint(20, 150),
        "clf__max_depth": randint(3, 15),
        "clf__learning_rate": loguniform(0.01, 0.3),
        "clf__n_estimators": randint(50, 500),
        "clf__min_child_samples": randint(2, 50),
        "clf__subsample": uniform(0.5, 0.5),
        "clf__colsample_bytree": uniform(0.5, 0.5),
        "clf__eg_alpha": loguniform(1e-3, 10),
        "clf__reg_lambda": loguniform(1e-3, 10),
        "clf__scale_pos_weight": uniform(1, 10),
    }

    grid_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="f1_weighted",
    )
    print(f"Ejecutando RandomizedSearchCV con 20 iteraciones sobre {len(X_train)} muestras...")
    grid_search.fit(X_train, y_train)
    print(f"Mejor score: {grid_search.best_score_:.4f}")
    print(f"Mejores parametros: {grid_search.best_params_}")

    return grid_search
