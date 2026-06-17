"""
Genera Tabla 1 y Tabla 2 del estilo Babaei & Giudici (2024) a partir de
los resultados del experimento Informed GPT y los modelos ML entrenados.

Uso:
    uv run python generate_tables.py

Salida:
    plots/tabla1_comparacion_modelos.png
    plots/tabla2_informed_gpt_por_tamanio.png
    plots/tablas_combinadas.png
"""
import sys
import types
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ── Shim para modelos serializados con versión antigua de skrub ──────────────
if "skrub._to_float32" not in sys.modules:
    _mod = types.ModuleType("skrub._to_float32")
    from sklearn.base import BaseEstimator, TransformerMixin
    class _ToFloat32(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None): return self
        def transform(self, X): return np.asarray(X, dtype=np.float32)
    _mod.ToFloat32 = _ToFloat32
    sys.modules["skrub._to_float32"] = _mod

from src.data import load_data

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "llm_tesis" / "results"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Modelos ML a incluir en Tabla 1 (nombre en archivo → etiqueta display)
ML_MODELS = {
    "propios": {
        "xgb_model":   "XGBoost",
        "lgb_model":   "LightGBM",
        "lgbm_skrub":  "LightGBM (skrub)",
        "hist_model":  "HistGradientBoosting",
        "rfc_model":   "Random Forest",
    },
    "jhquiza": {
        "best_pipe_xgb":     "XGBoost",
        "best_pipe_lgb":     "LightGBM",
        "best_pipe_lrc":     "Logistic Regression",
        "best_pipe_lrc_bal": "Logistic Regression (balanced)",
        "best_pipe_rfc":     "Random Forest",
        "best_pipe_hist":    "HistGradientBoosting",
        "best_pipe_mlp":     "MLP",
    },
}
MODEL_SOURCE = "propios"


def load_ml_aucs(X_test, y_test):
    """Evalúa todos los modelos ML y retorna {label: auc}."""
    models_dir = BASE_DIR / "models" / MODEL_SOURCE
    results = {}
    for stem, label in ML_MODELS[MODEL_SOURCE].items():
        path = models_dir / f"{stem}.joblib"
        if not path.exists():
            print(f"  {stem} no encontrado, omitiendo")
            continue
        try:
            model = joblib.load(path)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
            else:
                print(f"  {stem}: sin predict_proba, omitido")
                continue
            auc = float(roc_auc_score(y_test, y_prob))
            results[label] = auc
            print(f"  {label}: AUC = {auc:.4f}")
        except Exception as e:
            print(f"  {stem} omitido: {e}")
    return results


def compute_stats(series):
    """Min, Max, Mean, Std de una serie de AUC."""
    return {
        "Min":  series.min(),
        "Max":  series.max(),
        "Mean": series.mean(),
        "Std":  series.std() if len(series) > 1 else float("nan"),
    }


def render_table(ax, data, title, subtitle=""):
    """
    Dibuja una tabla de publicación (estilo paper) en los ejes dados.
    data: lista de dicts con keys [Method/Example size, Min, Max, Mean, Std]
    """
    ax.axis("off")
    columns = list(data[0].keys())
    cell_values = [[row[c] for c in columns] for row in data]

    table = ax.table(
        cellText=cell_values,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Estilo encabezado
    for j, col in enumerate(columns):
        cell = table[0, j]
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(fontweight="bold")

    # Líneas horizontales (estilo paper: solo arriba y abajo de encabezado + fondo)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_linewidth(1.2)

    ax.set_title(f"{title}\n{subtitle}", fontsize=10, fontweight="bold",
                 loc="left", pad=8)


def fmt(val, n_dec=10):
    """Formatea float con máximos decimales del paper; NaN → '-'."""
    if isinstance(val, float) and np.isnan(val):
        return "—"
    return f"{val:.{n_dec}f}"


def main():
    print("Cargando datos...")
    X_train, X_test, y_train, y_test = load_data()
    y_test_arr = y_test.values if hasattr(y_test, "values") else np.array(y_test)
    n_test = len(y_test_arr)
    print(f"Test set: {n_test} observaciones")

    # ── GPT results ──────────────────────────────────────────────────────────
    results_path = RESULTS_DIR / "experiment_results.csv"
    if not results_path.exists():
        print(f"ERROR: no se encontró {results_path}")
        print("Ejecuta primero: uv run python run.py")
        return

    df = pd.read_csv(results_path)
    n_runs_total = df["run"].nunique()
    print(f"Runs GPT encontrados: {n_runs_total} | Configuraciones: {sorted(df['n_ejemplos'].unique())}")
    if n_runs_total < 2:
        print("AVISO: solo hay 1 run por configuración. Std no será significativo.")
        print("Recomendación: ejecutar con informed_gpt_n_runs=5 en run.py para resultados completos.")

    zero_shot = df[df["n_ejemplos"] == 0]["auc_gpt"]
    informed = df[df["n_ejemplos"] > 0]
    example_sizes = sorted(informed["n_ejemplos"].unique())

    # AUC global de Informed GPT (todos los runs de todos los tamaños)
    informed_all_auc = informed["auc_gpt"]

    # ── ML AUCs ───────────────────────────────────────────────────────────────
    print("\nEvaluando modelos ML...")
    ml_aucs = load_ml_aucs(X_test, y_test_arr)

    # ═══════════════════════════════════════════════════════════════════════════
    # TABLA 1: Comparación de métodos
    # Informed GPT | GPT zero-shot | cada modelo ML
    # ═══════════════════════════════════════════════════════════════════════════
    tabla1_rows = []

    # Informed GPT (todos los tamaños combinados, promedio por run)
    # Para comparar con el paper: usamos todos los AUC de informed GPT
    if len(informed_all_auc) > 0:
        s = compute_stats(informed_all_auc)
        tabla1_rows.append({
            "Método": "Informed GPT",
            "Min":    fmt(s["Min"]),
            "Max":    fmt(s["Max"]),
            "Mean":   fmt(s["Mean"]),
            "Std":    fmt(s["Std"]),
        })

    # GPT zero-shot
    if len(zero_shot) > 0:
        s = compute_stats(zero_shot)
        tabla1_rows.append({
            "Método": "GPT (zero-shot)",
            "Min":    fmt(s["Min"]),
            "Max":    fmt(s["Max"]),
            "Mean":   fmt(s["Mean"]),
            "Std":    fmt(s["Std"]),
        })

    # Modelos ML
    for label, auc in sorted(ml_aucs.items(), key=lambda x: -x[1]):
        tabla1_rows.append({
            "Método": label,
            "Min":    fmt(auc),
            "Max":    fmt(auc),
            "Mean":   fmt(auc),
            "Std":    "—",
        })

    # ═══════════════════════════════════════════════════════════════════════════
    # TABLA 2: Informed GPT por tamaño de ejemplos
    # ═══════════════════════════════════════════════════════════════════════════
    tabla2_rows = []
    for n_ex in example_sizes:
        sub = df[df["n_ejemplos"] == n_ex]["auc_gpt"]
        s = compute_stats(sub)
        tabla2_rows.append({
            "Tamaño de ejemplos": int(n_ex),
            "Min":  fmt(s["Min"]),
            "Max":  fmt(s["Max"]),
            "Mean": fmt(s["Mean"]),
            "Std":  fmt(s["Std"]),
        })

    # ── Mostrar en consola ────────────────────────────────────────────────────
    print("\n=== TABLA 1: Comparación de métodos ===")
    print(pd.DataFrame(tabla1_rows).to_string(index=False))

    print("\n=== TABLA 2: Informed GPT por tamaño de ejemplos ===")
    print(pd.DataFrame(tabla2_rows).to_string(index=False))

    # ── Generar figura con ambas tablas ──────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 3 + 0.5 * max(len(tabla1_rows), 4)))
    fig.subplots_adjust(hspace=0.6)

    render_table(
        axes[0], tabla1_rows,
        title="Tabla 1",
        subtitle=f"Comparación de modelos según AUC. Test set: {n_test} observaciones. "
                 f"GPT: {n_runs_total} run(s).",
    )
    render_table(
        axes[1], tabla2_rows,
        title="Tabla 2",
        subtitle=f"AUC del modelo Informed GPT según tamaño del conjunto de ejemplos. "
                 f"Cada fila = {n_runs_total} run(s).",
    )

    out_path = PLOTS_DIR / "tablas_combinadas.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nFigura guardada en: {out_path}")

    # Tablas individuales también
    fig1, ax1 = plt.subplots(figsize=(13, 0.6 * len(tabla1_rows) + 2))
    render_table(ax1, tabla1_rows, "Tabla 1",
                 f"Comparación de modelos según AUC. Test set: {n_test} obs. GPT: {n_runs_total} run(s).")
    fig1.savefig(PLOTS_DIR / "tabla1_comparacion_modelos.png", bbox_inches="tight", dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(11, 0.6 * len(tabla2_rows) + 2))
    render_table(ax2, tabla2_rows, "Tabla 2",
                 f"AUC Informed GPT por tamaño de ejemplos. {n_runs_total} run(s) por fila.")
    fig2.savefig(PLOTS_DIR / "tabla2_informed_gpt_por_tamanio.png", bbox_inches="tight", dpi=150)
    plt.close(fig2)

    print("Tablas individuales guardadas en plots/")


if __name__ == "__main__":
    main()
