"""
Regenera todos los graficos de la tesis con labels en ingles
y actualiza tablas LaTeX con captions correctos.

Uso:  uv run python regenerate_plots.py
"""
import shutil
import sys
import types
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)

# ── Shim skrub ──
if "skrub._to_float32" not in sys.modules:
    _mod = types.ModuleType("skrub._to_float32")
    from sklearn.base import BaseEstimator, TransformerMixin
    class _ToFloat32(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None): return self
        def transform(self, X): return np.asarray(X, dtype=np.float32)
    _mod.ToFloat32 = _ToFloat32
    sys.modules["skrub._to_float32"] = _mod

from sklearn.model_selection import train_test_split

BASE_DIR   = Path(__file__).resolve().parent
PLOTS_DIR  = BASE_DIR / "plots"
DOC_DIR    = BASE_DIR / "document"
FIG_DIR    = DOC_DIR / "figuras"
TAB_DIR    = DOC_DIR / "tablas"
MODELS_DIR = BASE_DIR / "models" / "propios"
LLM_RESULTS = BASE_DIR / "llm_tesis" / "results"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY_LABELS = ["No default", "Default"]

ML_MODELS = {
    "xgb_model":   "XGBoost",
    "lgb_model":   "LightGBM",
    "lgbm_skrub":  "LightGBM (skrub)",
    "hist_model":  "HistGradientBoosting",
    "rfc_model":   "Random Forest",
}


def generate_comparative_roc(X_test, y_test, llm_y_true, llm_y_prob, df_preds=None):
    """ROC comparativa: todos los modelos ML + Informed GPT + LLM, labels en ingles."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # ML models
    ml_aucs = {}
    for stem, label in ML_MODELS.items():
        path = MODELS_DIR / f"{stem}.joblib"
        if not path.exists():
            print(f"  {stem} not found, skipping")
            continue
        model = joblib.load(path)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, linewidth=1.5, label=f"{label} (AUC={auc:.3f})")
        ml_aucs[label] = auc
        print(f"  {label}: AUC={auc:.4f}")

    # Informed GPT curves (promedio de runs por tamaño de ejemplos)
    if df_preds is not None and not df_preds.empty:
        linestyles = [":", "-.", (0, (3, 1, 1, 1)), (0, (5, 2))]
        for i, n_ex in enumerate(sorted(df_preds["n_ejemplos"].unique())):
            sub = df_preds[df_preds["n_ejemplos"] == n_ex].reset_index(drop=True)
            prob_cols = [c for c in sub.columns if c.startswith("y_prob_")]
            if not prob_cols:
                continue
            y_prob_mean = sub[prob_cols].mean(axis=1).values
            auc = roc_auc_score(y_test, y_prob_mean)
            fpr, tpr, _ = roc_curve(y_test, y_prob_mean)
            lbl = "GPT zero-shot" if n_ex == 0 else f"Informed GPT ({int(n_ex)} examples)"
            ls = linestyles[i % len(linestyles)]
            ax.plot(fpr, tpr, linewidth=2, linestyle=ls, label=f"{lbl} (AUC={auc:.3f})")
            print(f"  {lbl}: AUC={auc:.4f}")

    # LLM zero-shot (desde llm_predictions.csv — submuestra)
    fpr_llm, tpr_llm, _ = roc_curve(llm_y_true, llm_y_prob)
    auc_llm = roc_auc_score(llm_y_true, llm_y_prob)
    ax.plot(fpr_llm, tpr_llm, linewidth=2.5, linestyle="--",
            label=f"LLM (AUC={auc_llm:.3f})")
    print(f"  LLM: AUC={auc_llm:.4f}")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)

    out = PLOTS_DIR / "comparacion_roc_llm_vs_ml.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return ml_aucs


def generate_confusion_matrix_ml(X_test, y_test, model_stem, model_label):
    """Confusion matrix para un modelo ML, labels en ingles."""
    path = MODELS_DIR / f"{model_stem}.joblib"
    if not path.exists():
        print(f"  {model_stem} not found, skipping")
        return
    model = joblib.load(path)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=DISPLAY_LABELS, ax=ax
    )
    ax.set_title(model_label, fontsize=13)
    out = PLOTS_DIR / f"{model_stem}_confusion_matrix.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def generate_llm_confusion_matrix(y_true, y_pred):
    """Confusion matrix LLM, labels en ingles."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=DISPLAY_LABELS, ax=ax
    )
    ax.set_title("Confusion Matrix — LLM (GPT-4o-mini)", fontsize=13)
    out = PLOTS_DIR / "LLM_confusion_matrix.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def generate_llm_roc(y_true, y_prob):
    """ROC curve solo del LLM, labels en ingles."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"LLM (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — LLM (GPT-4o-mini)", fontsize=14)
    ax.legend()
    out = PLOTS_DIR / "LLM_roc_curve.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def generate_informed_gpt_confusion_matrices(df_preds, y_test):
    """Confusion matrix por tamaño de ejemplos (voto mayoritario entre runs)."""
    saved = []
    for n_ex in sorted(df_preds["n_ejemplos"].unique()):
        if n_ex == 0:
            continue
        sub = df_preds[df_preds["n_ejemplos"] == n_ex].reset_index(drop=True)
        pred_cols = [c for c in sub.columns if c.startswith("y_pred_")]
        if not pred_cols:
            continue
        y_pred_vote = (sub[pred_cols].mean(axis=1).values >= 0.5).astype(int)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred_vote, display_labels=DISPLAY_LABELS, ax=ax
        )
        ax.set_title(f"Informed GPT ({int(n_ex)} examples)", fontsize=13)
        out = PLOTS_DIR / f"informed_gpt_{int(n_ex)}_confusion_matrix.png"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved: {out.name}")
        saved.append((n_ex, out.name))
    return saved


def generate_tabla_ml(ml_aucs):
    rows = [{"Model": label, "AUC": auc}
            for label, auc in sorted(ml_aucs.items(), key=lambda x: -x[1])]
    df = pd.DataFrame(rows)
    df["AUC"] = df["AUC"].map(lambda x: f"{x:.4f}")

    tex = (
        "\\begin{table}[H]\n"
        "\\caption{AUC of ML models on the full test set.}\n"
        "\\label{tab:ml_auc}\n"
        "\\begin{tabular}{lc}\n"
        "\\toprule\n"
        "Model & AUC \\\\\n"
        "\\midrule\n"
    )
    for _, row in df.iterrows():
        tex += f"{row['Model']} & {row['AUC']} \\\\\n"
    tex += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    out = TAB_DIR / "tabla_ml.tex"
    out.write_text(tex)
    print(f"  Table saved: {out.name}")


def generate_tabla1(ml_aucs, df_gpt):
    rows = []

    informed = df_gpt[df_gpt["n_ejemplos"] > 0]["auc_gpt"]
    zero_shot = df_gpt[df_gpt["n_ejemplos"] == 0]["auc_gpt"]

    if len(informed) > 0:
        rows.append({
            "Method": "Informed GPT",
            "Min": f"{informed.min():.4f}",
            "Max": f"{informed.max():.4f}",
            "Mean": f"{informed.mean():.4f}",
            "Std": f"{informed.std():.4f}" if len(informed) > 1 else "---",
        })
    if len(zero_shot) > 0:
        rows.append({
            "Method": "GPT (zero-shot)",
            "Min": f"{zero_shot.min():.4f}",
            "Max": f"{zero_shot.max():.4f}",
            "Mean": f"{zero_shot.mean():.4f}",
            "Std": "---",
        })
    for label, auc in sorted(ml_aucs.items(), key=lambda x: -x[1]):
        rows.append({
            "Method": label,
            "Min": f"{auc:.4f}",
            "Max": f"{auc:.4f}",
            "Mean": f"{auc:.4f}",
            "Std": "---",
        })

    # Count actual runs per GPT method
    n_informed_runs = len(informed)
    n_total_configs = df_gpt[df_gpt["n_ejemplos"] > 0]["n_ejemplos"].nunique()

    tex = (
        "\\begin{table*}[!t]\n  \\small\n"
        "  \\centering\n"
        f"  \\caption{{Model comparison by AUC. For the Informed GPT, statistics are computed across all runs and example-size configurations. ML models are deterministic (single evaluation).}}\n"
        "  \\label{tab:comparacion}\n"
        "  \\begin{tabular}{lccc}\n"
        "    \\toprule\n"
        "    Method & Min & Max & Mean \\\\\n"
        "    \\midrule\n"
    )
    for row in rows:
        tex += f"    {row['Method']} & {row['Min']} & {row['Max']} & {row['Mean']} \\\\\n"
    tex += (
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table*}\n"
    )
    out = TAB_DIR / "tabla1_comparacion.tex"
    out.write_text(tex)
    print(f"  Table saved: {out.name}")


def generate_tabla2(df_gpt):
    informed = df_gpt[df_gpt["n_ejemplos"] > 0]
    example_sizes = sorted(informed["n_ejemplos"].unique())

    rows = []
    for n_ex in example_sizes:
        sub = informed[informed["n_ejemplos"] == n_ex]["auc_gpt"]
        n_runs = len(sub)
        rows.append({
            "Example size": int(n_ex),
            "Min": f"{sub.min():.4f}",
            "Max": f"{sub.max():.4f}",
            "Mean": f"{sub.mean():.4f}",
            "Std": f"{sub.std():.4f}" if n_runs > 1 else "---",
            "n_runs": n_runs,
        })

    # Build caption showing actual runs per row
    run_desc = ", ".join(f"$N={r['Example size']}$: {r['n_runs']}" for r in rows)
    caption = f"AUC of the Informed GPT model by example set size. Runs per config: {run_desc}."

    tex = (
        "\\begin{table*}[!t]\n  \\small\n"
        "  \\centering\n"
        f"  \\caption{{{caption}}}\n"
        "  \\label{tab:por_tamanio}\n"
        "  \\begin{tabular}{rcccc}\n"
        "    \\toprule\n"
        "    Example size & Min & Max & Mean & Std \\\\\n"
        "    \\midrule\n"
    )
    for row in rows:
        tex += f"    {row['Example size']} & {row['Min']} & {row['Max']} & {row['Mean']} & {row['Std']} \\\\\n"
    tex += (
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table*}\n"
    )
    out = TAB_DIR / "tabla2_tamanio.tex"
    out.write_text(tex)
    print(f"  Table saved: {out.name}")


def copy_to_figuras():
    """Copia plots regenerados a document/figuras/."""
    files = [
        "comparacion_roc_llm_vs_ml.png",
        "xgb_model_confusion_matrix.png",
        "lgb_model_confusion_matrix.png",
        "LLM_confusion_matrix.png",
        "LLM_roc_curve.png",
        "informed_gpt_10_confusion_matrix.png",
        "informed_gpt_20_confusion_matrix.png",
        "informed_gpt_40_confusion_matrix.png",
        "informed_gpt_80_confusion_matrix.png",
    ]
    for f in files:
        src = PLOTS_DIR / f
        if src.exists():
            shutil.copy2(src, FIG_DIR / f)
            print(f"  Copied to figuras/: {f}")


def main():
    print("Loading data (test_size=0.20 as per thesis)...")
    data_path = BASE_DIR / "base" / "WhatsApp Business Data (1).parquet"
    merged_df = pd.read_parquet(data_path)
    X = merged_df.drop("default", axis=1)
    cat_cols = X.select_dtypes(include="object").columns.to_list()
    X[cat_cols] = X[cat_cols].astype("category")
    y = merged_df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, stratify=y, random_state=1
    )
    y_test_arr = y_test.values
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    print("\nLoading LLM predictions...")
    llm_path = LLM_RESULTS / "llm_predictions.csv"
    df_llm = pd.read_csv(llm_path)
    llm_y_true = df_llm["y_true"].values
    llm_y_pred = df_llm["y_pred"].values
    llm_y_prob = df_llm["y_prob"].values
    print(f"  LLM predictions: {len(llm_y_true)} samples")

    print("\nLoading Informed GPT predictions...")
    preds_path = LLM_RESULTS / "experiment_preds.csv"
    df_preds = pd.read_csv(preds_path) if preds_path.exists() else None
    if df_preds is not None:
        print(f"  Informed GPT preds: {df_preds['n_ejemplos'].nunique()} configs")
    else:
        print("  No Informed GPT preds found, skipping")

    print("\nGenerating comparative ROC (English)...")
    ml_aucs = generate_comparative_roc(X_test, y_test_arr, llm_y_true, llm_y_prob, df_preds=df_preds)

    print("\nGenerating ML confusion matrices (English)...")
    generate_confusion_matrix_ml(X_test, y_test, "xgb_model", "XGBoost")
    generate_confusion_matrix_ml(X_test, y_test, "lgb_model", "LightGBM")

    print("\nGenerating LLM plots (English)...")
    # Zero-shot confusion matrix sobre el test set de 644 obs (mismo que ML e Informed GPT),
    # tomado de experiment_preds.csv n_ejemplos=0. Evita comparar peras con manzanas con el
    # archivo llm_predictions.csv (test set de 2573 del split 80/20 antiguo).
    if df_preds is not None and (df_preds["n_ejemplos"] == 0).any():
        n0 = df_preds[df_preds["n_ejemplos"] == 0].reset_index(drop=True)
        zs_pred = n0["y_pred_0"].astype(int).values
        zs_prob = n0["y_prob_0"].values
        generate_llm_confusion_matrix(y_test_arr, zs_pred)
        generate_llm_roc(y_test_arr, zs_prob)
    else:
        generate_llm_confusion_matrix(llm_y_true, llm_y_pred)
        generate_llm_roc(llm_y_true, llm_y_prob)

    print("\nGenerating Informed GPT confusion matrices...")
    informed_cms = []
    if df_preds is not None:
        informed_cms = generate_informed_gpt_confusion_matrices(df_preds, y_test_arr)

    print("\nCopying plots to document/figuras/...")
    copy_to_figuras()

    print("\nGenerating LaTeX tables...")
    generate_tabla_ml(ml_aucs)

    gpt_results = LLM_RESULTS / "experiment_results.csv"
    df_gpt = pd.read_csv(gpt_results)
    generate_tabla1(ml_aucs, df_gpt)
    generate_tabla2(df_gpt)

    print("\nDone! Files updated in:")
    print(f"  {FIG_DIR}")
    print(f"  {TAB_DIR}")


if __name__ == "__main__":
    main()
