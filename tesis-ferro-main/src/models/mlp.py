import warnings

import numpy as np
from prefect import task
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skrub import TableVectorizer

warnings.filterwarnings("ignore")


@task(name="entrenar_mlp", log_prints=True)
def train(X_train, y_train):
    print("Configurando MLPClassifier + RandomizedSearchCV...")
    pipe = Pipeline([
        ("prep", TableVectorizer()),
        ("scaler", StandardScaler(with_mean=False)),
        ("classifier", MLPClassifier(
            learning_rate_init=0.001,
            hidden_layer_sizes=[10],
            activation="logistic",
        )),
    ])

    param_distributions = {
        "classifier__max_iter": [1000],
        "classifier__alpha": 10.0 ** -np.arange(-3, 2),
        "classifier__learning_rate_init": loguniform(0.001, 0.1),
        "classifier__hidden_layer_sizes": [
            (10, 5), (20, 10), (20,), (50, 20), (100, 20), (100,)
        ],
        "classifier__activation": ["logistic", "tanh", "relu"],
    }

    grid_search_mlp = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=50,
        scoring="f1_weighted",
    )
    print(f"Ejecutando RandomizedSearchCV con 50 iteraciones sobre {len(X_train)} muestras...")
    grid_search_mlp.fit(X_train, y_train)
    print(f"Mejor score: {grid_search_mlp.best_score_:.4f}")
    print(f"Mejores parametros: {grid_search_mlp.best_params_}")

    return grid_search_mlp
