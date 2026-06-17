"""
Zero-shot LLM — clasificacion sin ejemplos en el prompt.

El modelo recibe las variables del cliente y el contexto del sector solidario
colombiano, pero ningun ejemplo real del dataset. Predice solo con conocimiento
previo del LLM.
"""
from prefect import flow

from llm_tesis.evaluate import (
    plot_comparative_roc,
    plot_confusion_matrix,
    plot_roc_curve,
    print_classification_report,
)
from llm_tesis.predict import load_results, predict_batch, save_results, stratified_subset


@flow(name="Zero-shot LLM — Prediccion de Default", log_prints=True)
def zero_shot_flow(
    X_test,
    y_test,
    n_samples=200,
    model="gpt-4o-mini",
    skip_predict=False,
    results_file="llm_predictions.csv",
):
    """
    Clasifica clientes usando LLM sin ejemplos en el prompt.

    Parametros:
        n_samples    : cuantos clientes del test set clasificar (None = todos)
        model        : modelo OpenAI a usar
        skip_predict : True = cargar resultados del CSV sin llamar a la API
        results_file : nombre del CSV donde se guardan/leen los resultados
    """
    print(f"Zero-shot LLM | modelo={model} | n_samples={n_samples} | cache={skip_predict}")

    X_sub, y_sub = stratified_subset(X_test, y_test, n_samples)

    if skip_predict:
        y_true, y_pred, y_prob = load_results(results_file)
    else:
        y_pred, y_prob, reasonings = predict_batch(X_sub, model=model)
        y_true = y_sub.values if hasattr(y_sub, "values") else y_sub
        save_results(X_sub, y_sub, y_pred, y_prob, reasonings, filename=results_file)

    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, model_name="Zero-shot LLM")
    auc = plot_roc_curve(y_true, y_prob, model_name="Zero-shot LLM")
    plot_comparative_roc(y_true, y_prob, X_sub)

    print(f"Zero-shot LLM finalizado | AUC={auc:.3f}")
    return auc
