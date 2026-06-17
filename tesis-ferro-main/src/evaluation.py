import base64
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from prefect import task
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


def _fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _sanitize_key(name):
    return name.lower().replace("_", "-")


@task(name="evaluar_modelo_ml", log_prints=True)
def evaluate(model, X_test, y_test, model_name="model", labels=None):
    display_labels = ["No default", "Default"]
    target_names = display_labels
    artifact_key = _sanitize_key(model_name)

    if labels is not None:
        target_names = labels
        display_labels = labels

    print(f"Generando predicciones para {model_name}...")
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=target_names)
    print(f"Reporte de clasificacion para {model_name}:")
    print(report)

    report_dict = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True
    )
    table = []
    for label in target_names + ["macro avg", "weighted avg"]:
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
        key=f"reporte-clasificacion-{artifact_key}",
        table=table,
        description=f"Reporte de clasificacion del modelo {model_name}",
    )

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=display_labels, ax=ax
    )
    ax.set_title(model_name)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / f"{model_name}_confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight")

    img_b64 = _fig_to_base64(fig)
    plt.close(fig)

    create_markdown_artifact(
        key=f"matriz-confusion-{artifact_key}",
        markdown=f"## Matriz de Confusion - {model_name}\n\n![Matriz de Confusion](data:image/png;base64,{img_b64})",
        description=f"Matriz de confusion para el modelo {model_name}",
    )

    print(f"Matriz de confusion guardada: {path}")
    print(f"Artifacts creados: reporte-clasificacion-{artifact_key}, matriz-confusion-{artifact_key}")
