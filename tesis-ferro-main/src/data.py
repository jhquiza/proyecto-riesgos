from pathlib import Path

import pandas as pd
from prefect import task
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "base"
DATA_FILE = "WhatsApp Business Data (1).parquet"


@task(name="cargar_datos", log_prints=True)
def load_data():
    path = DATA_DIR / DATA_FILE
    print(f"Leyendo archivo: {path}")
    merged_df = pd.read_parquet(path)
    print(f"Dataset cargado: {merged_df.shape[0]} filas, {merged_df.shape[1]} columnas")

    X = merged_df.drop("default", axis=1)

    # The type of categorical variables is changed to 'category' so that the
    # models can handle them correctly.
    categorical_features = X.select_dtypes(include="object").columns.to_list()
    X[categorical_features] = X[categorical_features].astype("category")
    print(f"Features categoricas convertidas: {categorical_features}")

    y = merged_df["default"]
    print(f"Distribucion target - No default: {(y == 0).sum()}, Default: {(y == 1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, stratify=y, random_state=1
    )
    print(f"Split realizado - Train: {len(X_train)}, Test: {len(X_test)}")

    return X_train, X_test, y_train, y_test
