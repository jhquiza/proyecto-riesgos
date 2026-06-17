import warnings

from prefect import task
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from skrub import TableVectorizer

warnings.filterwarnings("ignore")


@task(name="entrenar_random_forest", log_prints=True)
def train(X_train, y_train):
    print("Configurando RandomForestClassifier + RandomizedSearchCV...")
    rfc_model = RandomForestClassifier(class_weight="balanced")

    pipe = Pipeline([
        ("prep", TableVectorizer()),
        ("classifier", rfc_model),
    ])

    param_distributions = {
        "classifier__n_estimators": randint(50, 500),
        "classifier__max_depth": randint(3, 15),
        "classifier__min_samples_split": randint(2, 20),
        "classifier__min_samples_leaf": randint(1, 20),
        "classifier__max_features": ["auto", "sqrt", "log2", None],
        "classifier__bootstrap": [True, False],
        "classifier__criterion": ["gini", "entropy", "log_loss"],
    }

    grid_search_rfc = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=50,
        scoring="f1_weighted",
    )
    print(f"Ejecutando RandomizedSearchCV con 50 iteraciones sobre {len(X_train)} muestras...")
    grid_search_rfc.fit(X_train, y_train)
    print(f"Mejor score: {grid_search_rfc.best_score_:.4f}")
    print(f"Mejores parametros: {grid_search_rfc.best_params_}")

    return grid_search_rfc
