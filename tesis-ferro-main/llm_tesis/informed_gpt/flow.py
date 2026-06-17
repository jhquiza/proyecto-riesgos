"""
Informed GPT — replica adaptada del experimento de Babaei & Giudici (2024).

Referencia:
    Babaei & Giudici (2024). "GPT classifications, with application to credit lending."
    Machine Learning with Applications, 16, 100534.

Diferencias vs el paper original:
    - Test set: COMPLETO (2573 obs fijas) en lugar de 170 muestras aleatorias por run.
      El split 80/20 es siempre el mismo (random_state=1 en load_data).
    - Modelos ML: TODOS los .joblib en models_dir, no solo Logistic Regression.
    - ML AUC: valor fijo y deterministico (mismo test set siempre) — sin varianza.
    - Runs: aplican SOLO al GPT (distintas muestras de few-shot del train por run).
"""
import base64
import sys
import types
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from llm_tesis.predict import predict_batch

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Shim: los modelos de jhquiza fueron serializados con una version de skrub
# que tenia skrub._to_float32.ToFloat32.
if "skrub._to_float32" not in sys.modules:
    _mod = types.ModuleType("skrub._to_float32")
    class _ToFloat32(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None): return self
        def transform(self, X): return np.asarray(X, dtype=np.float32)
    _mod.ToFloat32 = _ToFloat32
    sys.modules["skrub._to_float32"] = _mod

DISPLAY_LABELS = ["No default", "Default"]


def _fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def _load_all_models(models_dir):
    """Carga todos los .joblib del directorio. Retorna {stem: model}."""
    models = {}
    for path in sorted(Path(models_dir).glob("*.joblib")):
        try:
            models[path.stem] = joblib.load(path)
            print(f"  Modelo ML cargado: {path.name}")
        except Exception as e:
            print(f"  Modelo ML omitido ({path.name}): {e}")
    return models


def _compute_ml_aucs(ml_models, X_test, y_test_arr):
    """
    Evalua todos los modelos ML sobre el test set completo.
    Retorna {stem: {"auc": float, "y_prob": array, "y_pred": array} | None}.
    """
    results = {}
    for stem, model in ml_models.items():
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
            else:
                print(f"  {stem}: sin predict_proba ni decision_function, omitido")
                results[stem] = None
                continue
            y_pred = (y_prob >= 0.5).astype(int)
            auc = float(roc_auc_score(y_test_arr, y_prob))
            results[stem] = {"auc": auc, "y_prob": y_prob, "y_pred": y_pred}
            print(f"  AUC {stem} = {auc:.4f}")
        except Exception as e:
            print(f"  {stem} omitido: {e}")
            results[stem] = None
    return results


def _sample_train_examples(X_train, y_train, n_ex, run):
    """Muestrea n_ex ejemplos estratificados del train. Semilla = run + 100."""
    _, X_ex, _, y_ex = train_test_split(
        X_train, y_train,
        test_size=n_ex,
        stratify=y_train,
        random_state=run + 100,
    )
    return X_ex, y_ex


@task(name="publicar_resultados_ml", log_prints=True)
def _publish_ml_results(ml_results, n_test):
    """Publica AUCs de ML inmediatamente, sin esperar al LLM."""
    validos = {s: r for s, r in ml_results.items() if r is not None}
    if not validos:
        return
    tabla = [
        {"Modelo": stem, "AUC": f"{r['auc']:.4f}", "N test": n_test}
        for stem, r in sorted(validos.items(), key=lambda x: -x[1]["auc"])
    ]
    create_table_artifact(
        key="ml-resultados",
        table=tabla,
        description=f"AUC modelos ML sobre test completo ({n_test} obs) — deterministico",
    )
    mejor = max(validos, key=lambda s: validos[s]["auc"])
    print(f"ML publicado: mejor={mejor} AUC={validos[mejor]['auc']:.4f}")


def _gpt_label(n_ex):
    return "GPT zero-shot" if n_ex == 0 else f"Informed GPT ({n_ex} ejemplos)"


def _model_section(label, y_true, y_prob, y_pred, auc, runs_info=""):
    """Genera ROC individual + confusion matrix para un modelo. Retorna markdown."""
    # ROC individual
    fig_roc, ax = plt.subplots(figsize=(5, 4))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax.plot(fpr, tpr, lw=2, color="#2980B9")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title(f"ROC — {label} (AUC={auc:.4f})")
    roc_b64 = _fig_to_b64(fig_roc)

    # Confusion matrix
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=DISPLAY_LABELS, ax=ax_cm
    )
    ax_cm.set_title(label)
    cm_b64 = _fig_to_b64(fig_cm)

    return f"""
### {label}
AUC: **{auc:.4f}**{runs_info}

| | ROC | Confusion Matrix |
|---|---|---|
| | ![ROC](data:image/png;base64,{roc_b64}) | ![CM](data:image/png;base64,{cm_b64}) |

---
"""


@task(name="publicar_resultados_informed_gpt", log_prints=True)
def _publish_results(df_gpt, ml_results, example_sizes, y_test_arr, n_test):
    """
    Genera por cada modelo (GPT n_ex + ML joblib):
      - ROC curve individual
      - Confusion matrix individual
    Mas una ROC comparativa con todos.
    """
    ml_validos = {s: r for s, r in ml_results.items() if r is not None}
    gpt_preds_path = RESULTS_DIR / "experiment_preds.csv"
    df_preds = pd.read_csv(gpt_preds_path) if gpt_preds_path.exists() else pd.DataFrame()

    # ── Tabla ranking ──────────────────────────────────────────────────────────
    tabla = []
    for n_ex in example_sizes:
        sub = df_gpt[df_gpt["n_ejemplos"] == n_ex]["auc_gpt"]
        tabla.append({
            "Metodo": _gpt_label(n_ex), "Tipo": "LLM", "N test": n_test,
            "Mean AUC": f"{sub.mean():.4f}",
            "Std AUC": f"{sub.std():.4f}" if len(sub) > 1 else "—",
            "Min AUC": f"{sub.min():.4f}", "Max AUC": f"{sub.max():.4f}",
            "Runs": int(len(sub)),
        })
    for stem, r in sorted(ml_validos.items(), key=lambda x: -x[1]["auc"]):
        tabla.append({
            "Metodo": stem, "Tipo": "ML", "N test": n_test,
            "Mean AUC": f"{r['auc']:.4f}", "Std AUC": "—",
            "Min AUC": f"{r['auc']:.4f}", "Max AUC": f"{r['auc']:.4f}", "Runs": 1,
        })
    tabla.sort(key=lambda r: float(r["Mean AUC"]), reverse=True)
    create_table_artifact(
        key="informed-gpt-detalle",
        table=tabla,
        description=f"Ranking AUC — test completo ({n_test} obs) — GPT y ML",
    )

    # ── ROC comparativa (todos en un solo grafico) ─────────────────────────────
    fig_roc_all, ax_all = plt.subplots(figsize=(11, 8))
    ax_all.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Aleatorio")
    for stem, r in ml_validos.items():
        fpr, tpr, _ = roc_curve(y_test_arr, r["y_prob"])
        ax_all.plot(fpr, tpr, lw=1.5, label=f"{stem} (AUC={r['auc']:.3f})")
    gpt_probs = {}
    if not df_preds.empty:
        for n_ex in example_sizes:
            sub = df_preds[df_preds["n_ejemplos"] == n_ex]
            prob_cols = [c for c in sub.columns if c.startswith("y_prob_")]
            if not prob_cols or len(sub) == 0:
                continue
            y_prob_mean = sub[prob_cols].mean(axis=1).values
            gpt_probs[n_ex] = y_prob_mean
            auc = df_gpt[df_gpt["n_ejemplos"] == n_ex]["auc_gpt"].mean()
            fpr, tpr, _ = roc_curve(y_test_arr, y_prob_mean)
            ax_all.plot(fpr, tpr, lw=2.5, ls="--", label=f"{_gpt_label(n_ex)} (AUC={auc:.3f})")
    ax_all.set_xlabel("Tasa de Falsos Positivos")
    ax_all.set_ylabel("Tasa de Verdaderos Positivos")
    ax_all.set_title(f"Curvas ROC Comparativas — todos los modelos | {n_test} obs")
    ax_all.legend(loc="lower right", fontsize=7)
    roc_all_b64 = _fig_to_b64(fig_roc_all)

    # ── Secciones individuales por modelo ─────────────────────────────────────
    secciones = ""

    # GPT configs
    for n_ex in example_sizes:
        if n_ex not in gpt_probs:
            continue
        sub_preds = df_preds[df_preds["n_ejemplos"] == n_ex]
        pred_cols = [c for c in sub_preds.columns if c.startswith("y_pred_")]
        y_pred_vote = (sub_preds[pred_cols].mean(axis=1).values >= 0.5).astype(int)
        auc = df_gpt[df_gpt["n_ejemplos"] == n_ex]["auc_gpt"].mean()
        n_runs = df_gpt[df_gpt["n_ejemplos"] == n_ex]["auc_gpt"].count()
        std = df_gpt[df_gpt["n_ejemplos"] == n_ex]["auc_gpt"].std()
        runs_info = f" | {n_runs} run(s)" + (f", Std={std:.4f}" if n_runs > 1 else "")
        secciones += _model_section(_gpt_label(n_ex), y_test_arr, gpt_probs[n_ex], y_pred_vote, auc, runs_info)

    # ML models
    for stem, r in sorted(ml_validos.items(), key=lambda x: -x[1]["auc"]):
        secciones += _model_section(f"{stem} (ML)", y_test_arr, r["y_prob"], r["y_pred"], r["auc"])

    # ── Markdown final ─────────────────────────────────────────────────────────
    best_ml = max(ml_validos, key=lambda s: ml_validos[s]["auc"]) if ml_validos else None
    best_n = df_gpt[df_gpt["n_ejemplos"] > 0].groupby("n_ejemplos")["auc_gpt"].mean().idxmax() if any(n > 0 for n in example_sizes) else example_sizes[0]
    best_gpt_auc = df_gpt[df_gpt["n_ejemplos"] == best_n]["auc_gpt"].mean()
    zero_auc = df_gpt[df_gpt["n_ejemplos"] == 0]["auc_gpt"].mean() if 0 in example_sizes else None

    md = f"""## Informed GPT — Resultados sobre test set completo ({n_test} obs)

**Referencia**: Babaei & Giudici (2024). *GPT classifications, with application to credit lending.*

> Split fijo (random_state=1). ML evaluado una vez (deterministico). GPT: {df_gpt['run'].nunique()} run(s).

## ROC Comparativa — Todos los Modelos

![ROC Comparativa](data:image/png;base64,{roc_all_b64})

---

## Detalle por Modelo

{secciones}
"""
    create_markdown_artifact(
        key="informed-gpt-analisis",
        markdown=md,
        description=f"ROC + CM por modelo | {len(example_sizes)} GPT configs + {len(ml_validos)} ML | AUC mejor GPT={best_gpt_auc:.4f}",
    )
    print(f"Artifacts publicados: informed-gpt-detalle, informed-gpt-analisis ({len(example_sizes) + len(ml_validos)} modelos)")


@flow(name="Informed GPT — Babaei-Giudici (2024)", log_prints=True)
def informed_gpt_flow(
    X_train,
    X_test,
    y_train,
    y_test,
    model="gpt-4o-mini",
    n_runs=5,
    example_sizes=None,
    results_file="experiment_results.csv",
    models_dir=None,
):
    """
    Replica adaptada de Babaei & Giudici (2024):
    - Split train/test FIJO (random_state=1 en load_data).
    - Test set COMPLETO (2573 obs), sin muestreo aleatorio.
    - ML AUC: deterministico, calculado UNA SOLA VEZ.
    - GPT: n_runs repeticiones con distintos ejemplos del train.
    - Al final: ROC curves + confusion matrices para GPT y ML.
    """
    if example_sizes is None:
        example_sizes = [0, 40, 80]

    if models_dir is None:
        models_dir = Path(__file__).resolve().parent.parent.parent / "models" / "propios"
    models_dir = Path(models_dir)

    n_test = len(X_test)
    total_llamadas = len(example_sizes) * n_runs * n_test
    print(f"Informed GPT | modelo={model} | {len(example_sizes)} config x {n_runs} runs | ~{total_llamadas} llamadas API")
    print(f"Test set: {n_test} obs (completo, deterministico) | modelos ML: {models_dir.name}")

    # ── ML: una sola vez ───────────────────────────────────────────────────────
    print("Evaluando modelos ML sobre test completo...")
    ml_models = _load_all_models(models_dir)
    y_test_arr = y_test.values if hasattr(y_test, "values") else np.array(y_test)
    ml_results = _compute_ml_aucs(ml_models, X_test, y_test_arr)
    ml_validos = {s: r for s, r in ml_results.items() if r is not None}
    print(f"Modelos ML con AUC: {len(ml_validos)}")
    _publish_ml_results(ml_results, n_test)

    # ── CSV de resultados GPT ──────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / results_file
    preds_path = RESULTS_DIR / "experiment_preds.csv"
    expected_cols = {"n_ejemplos", "run", "auc_gpt", "n_test"}

    if results_path.exists():
        df_cache = pd.read_csv(results_path)
        cached_n_test = int(df_cache["n_test"].iloc[0]) if len(df_cache) > 0 and "n_test" in df_cache.columns else None
        if not expected_cols.issubset(set(df_cache.columns)) or len(df_cache.columns) > 4 or cached_n_test != n_test:
            print(f"Cache incompatible (n_test cache={cached_n_test} vs actual={n_test}), reiniciando.")
            df_cache = pd.DataFrame(columns=["n_ejemplos", "run", "auc_gpt", "n_test"])
            done = set()
            if preds_path.exists():
                preds_path.unlink()
        else:
            done = set(zip(df_cache["n_ejemplos"].astype(int), df_cache["run"].astype(int)))
            print(f"Retomando: {len(done)} iteraciones ya completadas")
    else:
        df_cache = pd.DataFrame(columns=["n_ejemplos", "run", "auc_gpt", "n_test"])
        done = set()

    # CSV de predicciones para ROC y confusion matrix
    if preds_path.exists():
        df_preds = pd.read_csv(preds_path)
    else:
        df_preds = pd.DataFrame()

    total_iter = len(example_sizes) * n_runs
    print(f"Iteraciones GPT pendientes: {total_iter - len(done)}/{total_iter}")

    for n_ex in example_sizes:
        for run in range(n_runs):
            if (n_ex, run) in done:
                print(f"  [n_ejemplos={n_ex}, run={run}] ya completado, omitiendo")
                continue

            tipo = "zero-shot" if n_ex == 0 else f"informed ({n_ex} ejemplos)"
            print(f"  [n_ejemplos={n_ex}, run={run}] {tipo} sobre {n_test} obs")

            X_ex = y_ex = None
            if n_ex > 0:
                X_ex, y_ex = _sample_train_examples(X_train, y_train, n_ex, run)

            y_pred, y_prob, _ = predict_batch(X_test, model=model, few_shot_X=X_ex, few_shot_y=y_ex)
            auc_gpt = float(roc_auc_score(y_test_arr, y_prob))
            print(f"    AUC GPT={auc_gpt:.4f}")

            # Guardar resultados
            nueva_fila = pd.DataFrame([{"n_ejemplos": n_ex, "run": run, "auc_gpt": auc_gpt, "n_test": n_test}])
            df_cache = pd.concat([df_cache, nueva_fila], ignore_index=True)
            df_cache.to_csv(results_path, index=False)

            # Guardar predicciones para plots
            nueva_pred = pd.DataFrame({
                "n_ejemplos": n_ex,
                f"y_pred_{run}": y_pred,
                f"y_prob_{run}": y_prob,
            })
            if df_preds.empty or "n_ejemplos" not in df_preds.columns:
                df_preds = nueva_pred
            else:
                existing = df_preds[df_preds["n_ejemplos"] == n_ex]
                if existing.empty:
                    df_preds = pd.concat([df_preds, nueva_pred], ignore_index=True)
                else:
                    df_preds.loc[df_preds["n_ejemplos"] == n_ex, f"y_pred_{run}"] = y_pred
                    df_preds.loc[df_preds["n_ejemplos"] == n_ex, f"y_prob_{run}"] = y_prob
            df_preds.to_csv(preds_path, index=False)

            done.add((n_ex, run))

    print(f"Experimento completado. Resultados en {results_path}")
    df_cache = pd.read_csv(results_path)
    _publish_results(df_cache, ml_results, example_sizes, y_test_arr, n_test)

    return df_cache
