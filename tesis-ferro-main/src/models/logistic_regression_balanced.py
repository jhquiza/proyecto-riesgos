import warnings

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from prefect import task
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from skrub import TableVectorizer

warnings.filterwarnings("ignore")


@task(name="entrenar_logistic_regression_balanced", log_prints=True)
def train(X_train, y_train):
    print("Configurando LogisticRegression + RandomUnderSampler + RandomizedSearchCV...")
    pipe = ImbPipeline([
        ("prep", TableVectorizer()),
        ("scaler", StandardScaler(with_mean=False)),
        ("undersampling", RandomUnderSampler(
            sampling_strategy="majority", random_state=1
        )),
        ("classifier", LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
        )),
    ])

    param_distributions = {
        "classifier__penalty": ["l1", "l2"],
        "classifier__C": loguniform(1e-3, 1e3),
    }

    grid_search_lrc_bal = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=50,
        scoring="f1_weighted",
    )
    print(f"Ejecutando RandomizedSearchCV con 50 iteraciones sobre {len(X_train)} muestras...")
    grid_search_lrc_bal.fit(X_train, y_train)
    print(f"Mejor score: {grid_search_lrc_bal.best_score_:.4f}")
    print(f"Mejores parametros: {grid_search_lrc_bal.best_params_}")

    return grid_search_lrc_bal
