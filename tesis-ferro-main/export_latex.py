"""
Exporta figuras y tablas al formato LaTeX listo para incluir en document/tesis.tex.

Uso:
    uv run python export_latex.py

Salida:
    document/figuras/   — copias de los PNGs del pipeline
    document/tablas/    — archivos .tex con tablas en formato booktabs
"""
import shutil
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ── Shim skrub ───────────────────────────────────────────────────────────────
if "skrub._to_float32" not in sys.modules:
    _mod = types.ModuleType("skrub._to_float32")
    from sklearn.base import BaseEstimator, TransformerMixin
    class _ToFloat32(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None): return self
        def transform(self, X): return np.asarray(X, dtype=np.float32)
    _mod.ToFloat32 = _ToFloat32
    sys.modules["skrub._to_float32"] = _mod

from src.data import load_data

BASE_DIR   = Path(__file__).resolve().parent
PLOTS_DIR  = BASE_DIR / "plots"
DOC_DIR    = BASE_DIR / "document"
FIG_DIR    = DOC_DIR / "figuras"
TAB_DIR    = DOC_DIR / "tablas"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

ML_MODELS = {
    "xgb_model":   "XGBoost",
    "lgb_model":   "LightGBM",
    "lgbm_skrub":  "LightGBM (skrub)",
    "hist_model":  "HistGradientBoosting",
    "rfc_model":   "Random Forest",
}
MODELS_DIR = BASE_DIR / "models" / "propios"


def copy_figures():
    figuras = [
        "resumen_roc_todos.png",
        "xgb_model_confusion_matrix.png",
        "lgb_model_confusion_matrix.png",
        "hist_model_confusion_matrix.png",
        "rfc_model_confusion_matrix.png",
        "lgbm_skrub_confusion_matrix.png",
        "LLM_confusion_matrix.png",
        "LLM_roc_curve.png",
        "comparacion_roc_llm_vs_ml.png",
        "tablas_combinadas.png",
        "tabla1_comparacion_modelos.png",
        "tabla2_informed_gpt_por_tamanio.png",
    ]
    for f in figuras:
        src = PLOTS_DIR / f
        if src.exists():
            shutil.copy2(src, FIG_DIR / f)
            print(f"  figura copiada: {f}")
        else:
            print(f"  figura no encontrada (omitida): {f}")


def load_ml_aucs(X_test, y_test):
    results = {}
    for stem, label in ML_MODELS.items():
        path = MODELS_DIR / f"{stem}.joblib"
        if not path.exists():
            continue
        try:
            model = joblib.load(path)
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_prob))
            results[label] = auc
        except Exception as e:
            print(f"  {stem} omitido: {e}")
    return results


def tabla_ml(ml_aucs):
    rows = [{"Model": label, "AUC": auc}
            for label, auc in sorted(ml_aucs.items(), key=lambda x: -x[1])]
    df = pd.DataFrame(rows)
    df["AUC"] = df["AUC"].map(lambda x: f"{x:.4f}")

    tex = df.to_latex(
        index=False,
        escape=True,
        column_format="lc",
        caption="AUC of ML models on the full test set.",
        label="tab:ml_auc",
    )
    # Reemplazar \begin{table} por \begin{table}[H] y agregar booktabs
    tex = tex.replace(r"\begin{table}", r"\begin{table}[H]")
    tex = tex.replace(r"\hline", r"\midrule", 1)
    tex = tex.replace(r"\hline", r"\bottomrule", 1)
    tex = r"\toprule" + "\n" + tex.split(r"\toprule")[-1] if r"\toprule" not in tex else tex

    out = TAB_DIR / "tabla_ml.tex"
    out.write_text(tex)
    print(f"  tabla guardada: {out.name}")


def tabla1_comparacion(ml_aucs, df_gpt):
    rows = []

    informed = df_gpt[df_gpt["n_ejemplos"] > 0]["auc_gpt"]
    zero_shot = df_gpt[df_gpt["n_ejemplos"] == 0]["auc_gpt"]

    def stats_row(label, series, is_ml=False):
        if is_ml:
            auc = series
            return {
                "Method": label,
                "Min": f"{auc:.4f}",
                "Max": f"{auc:.4f}",
                "Mean": f"{auc:.4f}",
                "Std": "—",
            }
        return {
            "Method": label,
            "Min":  f"{series.min():.4f}",
            "Max":  f"{series.max():.4f}",
            "Mean": f"{series.mean():.4f}",
            "Std":  f"{series.std():.4f}" if len(series) > 1 else "—",
        }

    if len(informed) > 0:
        rows.append(stats_row("Informed GPT", informed))
    if len(zero_shot) > 0:
        rows.append(stats_row("GPT (zero-shot)", zero_shot))
    for label, auc in sorted(ml_aucs.items(), key=lambda x: -x[1]):
        rows.append(stats_row(label, auc, is_ml=True))

    df = pd.DataFrame(rows)
    n_runs = df_gpt["run"].nunique()

    tex = (
        "\\begin{table*}[!t]\n  \\small\n"
        "  \\centering\n"
        f"  \\caption{{Model comparison by AUC. GPT: {n_runs} run(s) per configuration.}}\n"
        "  \\label{tab:comparacion}\n"
        "  \\begin{tabular}{lcccc}\n"
        "    \\toprule\n"
        "    Method & Min & Max & Mean & Std \\\\\n"
        "    \\midrule\n"
    )
    for _, row in df.iterrows():
        tex += f"    {row['Method']} & {row['Min']} & {row['Max']} & {row['Mean']} & {row['Std']} \\\\\n"
    tex += (
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table*}\n"
    )

    out = TAB_DIR / "tabla1_comparacion.tex"
    out.write_text(tex)
    print(f"  tabla guardada: {out.name}")


def tabla2_tamanio(df_gpt):
    informed = df_gpt[df_gpt["n_ejemplos"] > 0]
    example_sizes = sorted(informed["n_ejemplos"].unique())
    n_runs = df_gpt["run"].nunique()

    rows = []
    for n_ex in example_sizes:
        sub = informed[informed["n_ejemplos"] == n_ex]["auc_gpt"]
        rows.append({
            "Tamaño": int(n_ex),
            "Min":    f"{sub.min():.4f}",
            "Max":    f"{sub.max():.4f}",
            "Mean":   f"{sub.mean():.4f}",
            "Std":    f"{sub.std():.4f}" if len(sub) > 1 else "—",
        })

    tex = (
        "\\begin{table*}[!t]\n  \\small\n"
        "  \\centering\n"
        f"  \\caption{{AUC of the Informed GPT model by example set size. Each row = {n_runs} corrida(s).}}\n"
        "  \\label{tab:por_tamanio}\n"
        "  \\begin{tabular}{rcccc}\n"
        "    \\toprule\n"
        "    Example size & Min & Max & Mean & Std \\\\\n"
        "    \\midrule\n"
    )
    for row in rows:
        tex += f"    {row['Tamaño']} & {row['Min']} & {row['Max']} & {row['Mean']} & {row['Std']} \\\\\n"
    tex += (
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table*}\n"
    )

    out = TAB_DIR / "tabla2_tamanio.tex"
    out.write_text(tex)
    print(f"  tabla guardada: {out.name}")


def main():
    print("Cargando datos...")
    X_train, X_test, y_train, y_test = load_data()
    y_test_arr = y_test.values if hasattr(y_test, "values") else np.array(y_test)

    print("\nCopiando figuras...")
    copy_figures()

    print("\nEvaluando modelos ML...")
    ml_aucs = load_ml_aucs(X_test, y_test_arr)

    print("\nLeyendo resultados GPT...")
    results_path = BASE_DIR / "llm_tesis" / "results" / "experiment_results.csv"
    if not results_path.exists():
        print(f"ERROR: no se encontró {results_path}")
        return
    df_gpt = pd.read_csv(results_path)
    n_runs = df_gpt["run"].nunique()
    print(f"  Runs: {n_runs} | Configuraciones: {sorted(df_gpt['n_ejemplos'].unique())}")

    print("\nGenerando tablas LaTeX...")
    tabla_ml(ml_aucs)
    tabla1_comparacion(ml_aucs, df_gpt)
    tabla2_tamanio(df_gpt)

    print(f"\nListo. Archivos en:")
    print(f"  {FIG_DIR}")
    print(f"  {TAB_DIR}")
    print(f"\nPara compilar en local:")
    print(f"  cd document && pdflatex tesis.tex && biber tesis && pdflatex tesis.tex")
    print(f"\nO sube la carpeta document/ a Overleaf.")


if __name__ == "__main__":
    main()
