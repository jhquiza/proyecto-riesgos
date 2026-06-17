import time
from pathlib import Path

import numpy as np
import pandas as pd
from prefect import task
from sklearn.model_selection import train_test_split

from llm_tesis.client import classify_row, create_client
from llm_tesis.prompt import build_few_shot_section

DEFAULT_MODEL = "gpt-4o-mini"

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@task(name="seleccionar_subconjunto_estratificado", log_prints=True)
def stratified_subset(X_test, y_test, n_samples):
    if n_samples is None or n_samples >= len(X_test):
        print(f"n_samples ({n_samples}) >= tamaño test ({len(X_test)}), usando todo el test set")
        return X_test, y_test
    X_sub, _, y_sub, _ = train_test_split(
        X_test, y_test,
        train_size=n_samples,
        stratify=y_test,
        random_state=42,
    )
    print(f"Subconjunto estratificado: {len(X_sub)} muestras")
    print(f"  No default: {(y_sub == 0).sum()} ({(y_sub == 0).mean() * 100:.1f}%)")
    print(f"  Default: {(y_sub == 1).sum()} ({(y_sub == 1).mean() * 100:.1f}%)")
    return X_sub, y_sub


@task(name="samplear_ejemplos_few_shot", log_prints=True)
def sample_few_shot_examples(X_train, y_train, n_examples, random_state=1):
    """Muestrea n_examples de forma estratificada del conjunto de entrenamiento."""
    _, X_ex, _, y_ex = train_test_split(
        X_train, y_train,
        test_size=n_examples,
        stratify=y_train,
        random_state=random_state,
    )
    print(f"Ejemplos few-shot: {len(X_ex)} ({(y_ex == 1).sum()} defaults, {(y_ex == 0).sum()} no-defaults)")
    return X_ex, y_ex


@task(name="predecir_batch_llm", log_prints=True)
def predict_batch(X, model=None, few_shot_X=None, few_shot_y=None):
    client = create_client()
    modelo = model or DEFAULT_MODEL

    few_shot_text = ""
    if few_shot_X is not None and few_shot_y is not None:
        few_shot_text = build_few_shot_section(few_shot_X, few_shot_y)
        print(f"Informed GPT: {len(few_shot_X)} ejemplos en el prompt")
    else:
        print("Zero-shot: sin ejemplos en el prompt")

    total = len(X)

    # Tokens por fila: ~1500 base + el few-shot en el prompt
    tokens_per_row = 1500 + len(few_shot_text) // 4
    TPM_LIMIT = 170_000  # 200k con margen de seguridad
    rows_per_min = TPM_LIMIT / tokens_per_row
    BATCH_SIZE = max(1, min(10, int(rows_per_min / 5)))
    BATCH_DELAY = max(6, int(60 * BATCH_SIZE * tokens_per_row / TPM_LIMIT) + 3)

    print(f"Enviando {total} filas a {modelo} | ~{tokens_per_row} tokens/fila | batch={BATCH_SIZE} | delay={BATCH_DELAY}s...")

    rows = [
        {k: v.item() if isinstance(v, np.generic) else v for k, v in X.iloc[idx].to_dict().items()}
        for idx in range(total)
    ]
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    results = []
    for i in range(0, total, BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch = rows[i:i + BATCH_SIZE]
        futures = [
            classify_row.submit(client, row, model=modelo, few_shot_text=few_shot_text)
            for row in batch
        ]
        results.extend(f.result() for f in futures)
        print(f"  Batch {batch_num}/{n_batches} completado ({min(i + BATCH_SIZE, total)}/{total})")
        if batch_num < n_batches:
            time.sleep(BATCH_DELAY)

    y_pred = np.array([r["prediction"] for r in results])
    y_prob = np.array([r["default_probability"] for r in results])
    reasonings = [r["reasoning"] for r in results]

    print(f"Predicciones completadas: {total} filas | defaults={( y_pred == 1).sum()} ({(y_pred == 1).mean() * 100:.1f}%) | prob_media={y_prob.mean():.3f}")
    return y_pred, y_prob, reasonings


@task(name="guardar_resultados_llm", log_prints=True)
def save_results(X, y_true, y_pred, y_prob, reasonings, filename="llm_predictions.csv"):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = X.reset_index(drop=True).copy()
    df["y_true"] = y_true.values if hasattr(y_true, "values") else y_true
    df["y_pred"] = y_pred
    df["y_prob"] = y_prob
    df["acierto"] = (df["y_true"] == df["y_pred"]).astype(int)
    df["reasoning"] = reasonings
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)

    accuracy = df["acierto"].mean() * 100
    print(f"Resultados guardados en {path}")
    print(f"  Total filas: {len(df)}")
    print(f"  Aciertos: {df['acierto'].sum()}/{len(df)} ({accuracy:.1f}%)")
    return path


@task(name="cargar_resultados_llm", log_prints=True)
def load_results(filename="llm_predictions.csv"):
    path = RESULTS_DIR / filename
    print(f"Cargando resultados desde {path}")
    df = pd.read_csv(path)
    print(f"  {len(df)} filas cargadas")
    return df["y_true"].values, df["y_pred"].values, df["y_prob"].values
