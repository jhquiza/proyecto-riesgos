import base64
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from prefect import task
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
)

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "propios"

DISPLAY_LABELS = ["No default", "Default"]


def _fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@task(name="reporte_clasificacion_llm", log_prints=True)
def print_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=DISPLAY_LABELS)
    print("Reporte de clasificacion LLM:")
    print(report)

    report_dict = classification_report(
        y_true, y_pred, target_names=DISPLAY_LABELS, output_dict=True
    )
    table = []
    for label in DISPLAY_LABELS + ["macro avg", "weighted avg"]:
        row = report_dict[label]
        table.append({
            "Clase": label,
            "Precision": f"{row['precision']:.3f}",
            "Recall": f"{row['recall']:.3f}",
            "F1-Score": f"{row['f1-score']:.3f}",
            "Support": int(row["support"]),
        })
    table.append({
        "Clase": "accuracy",
        "Precision": "",
        "Recall": "",
        "F1-Score": f"{report_dict['accuracy']:.3f}",
        "Support": int(report_dict['weighted avg']['support']),
    })

    create_table_artifact(
        key="reporte-clasificacion-llm",
        table=table,
        description="Reporte de clasificacion del modelo LLM",
    )
    print("Artifact creado: reporte-clasificacion-llm")


@task(name="graficar_matriz_confusion_llm", log_prints=True)
def plot_confusion_matrix(y_true, y_pred, model_name="LLM"):
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=DISPLAY_LABELS, ax=ax
    )
    ax.set_title(f"Matriz de Confusion - {model_name}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / f"{model_name}_confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight")

    img_b64 = _fig_to_base64(fig)
    plt.close(fig)

    create_markdown_artifact(
        key=f"matriz-confusion-{model_name.lower()}",
        markdown=f"## Matriz de Confusion - {model_name}\n\n![Matriz de Confusion](data:image/png;base64,{img_b64})",
        description=f"Matriz de confusion para el modelo {model_name}",
    )
    print(f"Matriz de confusion guardada: {path}")
    print(f"Artifact creado: matriz-confusion-{model_name.lower()}")


@task(name="graficar_roc_llm", log_prints=True)
def plot_roc_curve(y_true, y_prob, model_name="LLM"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Aleatorio")
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title(f"Curva ROC - {model_name}")
    ax.legend()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / f"{model_name}_roc_curve.png"
    fig.savefig(path, bbox_inches="tight")

    img_b64 = _fig_to_base64(fig)
    plt.close(fig)

    create_markdown_artifact(
        key=f"roc-{model_name.lower()}",
        markdown=f"## Curva ROC - {model_name}\n\n**AUC-ROC: {auc:.3f}**\n\n![Curva ROC](data:image/png;base64,{img_b64})",
        description=f"Curva ROC del modelo {model_name} (AUC={auc:.3f})",
    )
    print(f"Curva ROC guardada: {path}")
    print(f"AUC-ROC: {auc:.3f}")
    print(f"Artifact creado: roc-{model_name.lower()}")
    return auc


@task(name="graficar_roc_comparativa", log_prints=True)
def plot_comparative_roc(y_true, llm_prob, X_subset, ml_models=None):
    """Grafica curvas ROC comparativas entre el LLM y modelos ML."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Curva ROC del LLM
    fpr, tpr, _ = roc_curve(y_true, llm_prob)
    auc = roc_auc_score(y_true, llm_prob)
    ax.plot(fpr, tpr, linewidth=2, label=f"LLM (AUC={auc:.3f})")
    print(f"LLM AUC: {auc:.3f}")

    auc_table = [{"Modelo": "LLM", "AUC-ROC": f"{auc:.3f}"}]

    # Curvas ROC de modelos ML
    if ml_models is None:
        ml_models = {
            "lgb_model": "LightGBM",
            "xgb_model": "XGBoost",
            "hist_model": "HistGradientBoosting",
            "rfc_model": "RandomForest",
            "lrc_model": "LogisticRegression",
        }

    for model_file, model_label in ml_models.items():
        model_path = MODELS_DIR / f"{model_file}.joblib"
        if not model_path.exists():
            print(f"Modelo {model_file} no encontrado en {model_path}, omitiendo...")
            continue

        try:
            print(f"Cargando modelo {model_label} desde {model_path}...")
            model = joblib.load(model_path)
            if hasattr(model, "predict_proba"):
                ml_prob = model.predict_proba(X_subset)[:, 1]
            elif hasattr(model, "decision_function"):
                ml_prob = model.decision_function(X_subset)
            else:
                print(f"  {model_label}: no soporta probabilidades, omitiendo...")
                continue

            fpr_ml, tpr_ml, _ = roc_curve(y_true, ml_prob)
            auc_ml = roc_auc_score(y_true, ml_prob)
            ax.plot(fpr_ml, tpr_ml, linewidth=1.5, label=f"{model_label} (AUC={auc_ml:.3f})")
            auc_table.append({"Modelo": model_label, "AUC-ROC": f"{auc_ml:.3f}"})
            print(f"  {model_label} AUC: {auc_ml:.3f}")
        except Exception as e:
            print(f"  Error cargando {model_label}: {e}")

    ax.plot([0, 1], [0, 1], "k--", label="Aleatorio")
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title("Comparacion Curvas ROC: LLM vs Modelos ML")
    ax.legend(loc="lower right")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / "comparacion_roc_llm_vs_ml.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)

    img_b64 = _fig_to_base64(fig)
    plt.close(fig)

    create_markdown_artifact(
        key="roc-comparativa-llm-vs-ml",
        markdown=f"## Comparacion Curvas ROC: LLM vs Modelos ML\n\n![ROC Comparativa](data:image/png;base64,{img_b64})",
        description="Curvas ROC comparativas entre LLM y modelos ML",
    )

    create_table_artifact(
        key="comparacion-auc-llm-vs-ml",
        table=auc_table,
        description="Comparacion de AUC-ROC entre LLM y modelos ML",
    )

    print(f"Grafica comparativa guardada: {path}")
    print("Artifacts creados: roc-comparativa-llm-vs-ml, comparacion-auc-llm-vs-ml")
