import base64
import json
import sys
import types
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np

# Shim: los modelos de jhquiza fueron serializados con una version de skrub
# que tenia skrub._to_float32.ToFloat32. Se registra un stub para que pickle
# pueda reconstruir los objetos sin ese modulo.
if "skrub._to_float32" not in sys.modules:
    _mod = types.ModuleType("skrub._to_float32")
    from sklearn.base import BaseEstimator, TransformerMixin
    class _ToFloat32(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None): return self
        def transform(self, X): return np.asarray(X, dtype=np.float32)
    _mod.ToFloat32 = _ToFloat32
    sys.modules["skrub._to_float32"] = _mod
import matplotlib.pyplot as plt
import pandas as pd
from prefect import flow
from prefect.artifacts import create_markdown_artifact
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve

_BASE_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = _BASE_DIR / "plots"
LLM_RESULTS_DIR = _BASE_DIR / "llm_tesis" / "results"

# Labels por fuente de modelos
_MODEL_LABELS = {
    "propios": {
        "lgbm_skrub": "LightGBM (skrub) [propios]",
        "lgb_model":  "LightGBM (tuned) [propios]",
        "hist_model": "HistGradientBoosting [propios]",
        "xgb_model":  "XGBoost [propios]",
        "rfc_model":  "RandomForest [propios]",
        "lrc_model":  "LogisticRegression [propios]",
        "lrc_bal_model": "LogisticRegression (balanced) [propios]",
        "mlp_model":  "MLP [propios]",
    },
    "jhquiza": {
        "best_pipe_xgb":     "XGBoost [jhquiza]",
        "best_pipe_lgb":     "LightGBM [jhquiza]",
        "best_pipe_hist":    "HistGradientBoosting [jhquiza]",
        "best_pipe_rfc":     "RandomForest [jhquiza]",
        "best_pipe_lrc":     "LogisticRegression [jhquiza]",
        "best_pipe_lrc_bal": "LogisticRegression (balanced) [jhquiza]",
        "best_pipe_mlp":     "MLP [jhquiza]",
    },
}
# "reentrenar" guarda en propios → mismos labels
_MODEL_LABELS["reentrenar"] = _MODEL_LABELS["propios"]
TARGET_NAMES = ["No default", "Default"]


def _fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _confusion_matrix_b64(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=TARGET_NAMES, ax=ax
    )
    ax.set_title(title)
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def _extract_metrics(y_true, y_pred, y_prob=None):
    report = classification_report(y_true, y_pred, target_names=TARGET_NAMES, output_dict=True)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    return {
        "accuracy": report["accuracy"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "f1_default": report["Default"]["f1-score"],
        "precision_default": report["Default"]["precision"],
        "recall_default": report["Default"]["recall"],
        "support_default": int(report["Default"]["support"]),
        "n_samples": int(report["weighted avg"]["support"]),
        "auc": auc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


@flow(name="Resumen de Resultados", log_prints=True)
def summary_flow(X_test, y_test, models_dir=None, model_source="propios", llm_results_file="llm_predictions.csv"):
    if models_dir is None:
        models_dir = _BASE_DIR / "models" / "propios"
    models_dir = Path(models_dir)

    ml_model_labels = _MODEL_LABELS.get(model_source, _MODEL_LABELS["propios"])

    print("=== GENERANDO RESUMEN COMPARATIVO ===")
    print(f"    Fuente: '{model_source}' -> {models_dir}")

    entries = {}

    # --- Modelos ML ---
    for model_file, label in ml_model_labels.items():
        model_path = models_dir / f"{model_file}.joblib"
        if not model_path.exists():
            print(f"  Modelo {model_file} no encontrado, omitiendo...")
            continue

        print(f"  Evaluando {label}...")
        try:
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
        except Exception as e:
            print(f"  {model_file} omitido: {e}")
            continue

        params = None
        params_path = models_dir / f"{model_file}_params.json"
        if params_path.exists():
            with open(params_path) as f:
                params = json.load(f)

        entries[model_file] = {
            "label": label,
            "type": "ML",
            "metrics": _extract_metrics(y_test, y_pred, y_prob),
            "params": params,
        }

    # --- LLM ---
    llm_path = LLM_RESULTS_DIR / llm_results_file
    if llm_path.exists():
        print(f"  Cargando resultados LLM desde {llm_path}...")
        df_llm = pd.read_csv(llm_path)
        y_true_llm = df_llm["y_true"].values
        y_pred_llm = df_llm["y_pred"].values
        y_prob_llm = df_llm["y_prob"].values
        entries["llm"] = {
            "label": "LLM (gpt-4o-mini)",
            "type": "LLM",
            "metrics": _extract_metrics(y_true_llm, y_pred_llm, y_prob_llm),
            "params": None,
        }
        print(f"  LLM cargado: {len(y_true_llm)} muestras")
    else:
        print(f"  Sin resultados LLM en {llm_path}, omitiendo...")

    if not entries:
        print("No hay modelos ni resultados para resumir.")
        return

    # --- Tabla comparativa ---
    table_rows = []
    for entry in entries.values():
        m = entry["metrics"]
        table_rows.append({
            "Modelo": entry["label"],
            "Tipo": entry["type"],
            "N": str(m["n_samples"]),
            "Accuracy": f"{m['accuracy']:.3f}",
            "F1 Weighted": f"{m['f1_weighted']:.3f}",
            "F1 Default": f"{m['f1_default']:.3f}",
            "Precision Default": f"{m['precision_default']:.3f}",
            "Recall Default": f"{m['recall_default']:.3f}",
            "AUC-ROC": f"{m['auc']:.3f}" if m["auc"] is not None else "N/A",
        })
    table_rows.sort(key=lambda r: float(r["F1 Default"]), reverse=True)

    # --- ROC comparativa ---
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Aleatorio")
    for entry in entries.values():
        m = entry["metrics"]
        if m["y_prob"] is None or m["auc"] is None:
            continue
        fpr, tpr, _ = roc_curve(m["y_true"], m["y_prob"])
        lw = 2.5 if entry["type"] == "LLM" else 1.5
        ls = "--" if entry["type"] == "LLM" else "-"
        ax.plot(fpr, tpr, lw=lw, ls=ls, label=f"{entry['label']} (AUC={m['auc']:.3f})")
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title("Curvas ROC — Todos los Modelos")
    ax.legend(loc="lower right", fontsize=8)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "resumen_roc_todos.png", bbox_inches="tight", dpi=150)
    roc_b64 = _fig_to_base64(fig)
    plt.close(fig)

    # --- Mejor modelo (por F1 Default, solo los que tienen AUC) ---
    valid = {k: e for k, e in entries.items() if e["metrics"]["auc"] is not None}
    best_key = max(valid, key=lambda k: valid[k]["metrics"]["f1_default"])
    best = entries[best_key]
    bm = best["metrics"]

    # --- Markdown: ranking table ---
    md_table = (
        "| # | Modelo | Tipo | N | Accuracy | F1 Weighted | F1 Default"
        " | Precision Default | Recall Default | AUC-ROC |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    for i, row in enumerate(table_rows, 1):
        marker = " (mejor)" if row["Modelo"] == best["label"] else ""
        md_table += (
            f"| {i} | {row['Modelo']}{marker} | {row['Tipo']} | {row['N']}"
            f" | {row['Accuracy']} | {row['F1 Weighted']} | {row['F1 Default']}"
            f" | {row['Precision Default']} | {row['Recall Default']} | {row['AUC-ROC']} |\n"
        )

    # --- Markdown: secciones por modelo ML ---
    ml_sections = ""
    for key, entry in entries.items():
        if entry["type"] != "ML":
            continue
        m = entry["metrics"]
        auc_str = f"{m['auc']:.3f}" if m["auc"] is not None else "N/A"
        params_str = ""
        if entry["params"]:
            lines = "\n".join(f"  - `{k}`: {v}" for k, v in entry["params"].items())
            params_str = f"\n**Mejores hiperparametros:**\n{lines}\n"
        cm_b64 = _confusion_matrix_b64(m["y_true"], m["y_pred"], entry["label"])
        ml_sections += f"""
#### {entry['label']}
| Metrica | Valor |
|---|---|
| Accuracy | {m['accuracy']:.3f} |
| F1 Weighted | {m['f1_weighted']:.3f} |
| F1 Default | {m['f1_default']:.3f} |
| Precision Default | {m['precision_default']:.3f} |
| Recall Default | {m['recall_default']:.3f} |
| AUC-ROC | {auc_str} |
| N muestras | {m['n_samples']} |
{params_str}
![Matriz de Confusion](data:image/png;base64,{cm_b64})

---
"""

    # --- Markdown: seccion LLM ---
    llm_section = ""
    if "llm" in entries:
        lm = entries["llm"]["metrics"]
        llm_cm_b64 = _confusion_matrix_b64(lm["y_true"], lm["y_pred"], "LLM (gpt-4o-mini)")
        llm_section = f"""
## LLM — gpt-4o-mini

| Metrica | Valor |
|---|---|
| Accuracy | {lm['accuracy']:.3f} |
| F1 Weighted | {lm['f1_weighted']:.3f} |
| F1 Default | {lm['f1_default']:.3f} |
| Precision Default | {lm['precision_default']:.3f} |
| Recall Default | {lm['recall_default']:.3f} |
| AUC-ROC | {lm['auc']:.3f} |
| N muestras | {lm['n_samples']} (subconjunto estratificado) |

![Matriz de Confusion LLM](data:image/png;base64,{llm_cm_b64})

> El LLM fue evaluado sobre un subconjunto del test set, no sobre el set completo.
> Las metricas no son directamente comparables con los modelos ML.
"""

    markdown = f"""# Informe Comparativo — Prediccion de Default

## Ranking de Modelos

> Ordenado por **F1 Default** — metrica principal para deteccion de morosidad (clase minoritaria).

{md_table}

---

## Curvas ROC — Todos los Modelos

![ROC Comparativa](data:image/png;base64,{roc_b64})

---

## Recomendacion

**Mejor modelo: {best['label']}** (mayor F1 Default)

| Metrica | Valor |
|---|---|
| F1 Default | {bm['f1_default']:.3f} |
| Recall Default | {bm['recall_default']:.3f} |
| Precision Default | {bm['precision_default']:.3f} |
| AUC-ROC | {bm['auc']:.3f} |
| Accuracy | {bm['accuracy']:.3f} |
| N muestras | {bm['n_samples']} |

> **Recall Default** indica que fraccion de morosos reales captura el modelo.
> Alta prioridad para minimizar perdidas por clientes no detectados.

---

## Modelos ML — Detalle por Modelo

{ml_sections}
{llm_section}
"""

    create_markdown_artifact(
        key="informe-resumen-final",
        markdown=markdown,
        description=f"Informe completo — Mejor modelo: {best['label']} (F1 Default: {bm['f1_default']:.3f})",
    )
    print("  Artifact creado: informe-resumen-final")
    print(f"=== RESUMEN COMPLETADO — Mejor modelo: {best['label']} ===")
