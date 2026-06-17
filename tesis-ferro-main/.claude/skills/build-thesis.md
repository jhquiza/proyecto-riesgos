# build-thesis

Regenera los gráficos de la tesis con labels en inglés, actualiza tablas LaTeX, y compila el PDF.

## Instructions

1. Regenerar plots y tablas ejecutando:
```bash
uv run python regenerate_plots.py
```

Esto genera:
- `plots/comparacion_roc_llm_vs_ml.png` — ROC comparativa (todos los ML + LLM)
- `plots/xgb_model_confusion_matrix.png` — Confusion matrix XGBoost
- `plots/lgb_model_confusion_matrix.png` — Confusion matrix LightGBM
- `plots/LLM_confusion_matrix.png` — Confusion matrix LLM
- `plots/LLM_roc_curve.png` — ROC curve LLM solo

Y copia todo a `document/figuras/`, regenera tablas en `document/tablas/`.

2. Compilar el PDF con tectonic:
```bash
cd document && tectonic tesis.tex
```

3. Verificar el PDF resultante en `document/tesis.pdf`.

## Notas importantes

- **Compilador LaTeX**: usar siempre `tectonic`, NO pdflatex/MacTeX (no están instalados)
- **Test split**: `regenerate_plots.py` usa `test_size=0.20` (2,573 obs) para coincidir con la tesis. `src/data.py` usa `test_size=0.05` para el pipeline de experimentación — son splits distintos a propósito
- **Labels**: todos los plots se generan con labels en **inglés** (el paper está en inglés)
- **Datos fuente**: modelos ML en `models/propios/*.joblib`, predicciones LLM en `llm_tesis/results/`
- **LaTeX source**: `document/tesis.tex` + `document/referencias.bib` + tablas en `document/tablas/`
- Los gráficos que usa el PDF están en `document/figuras/`, NO directamente desde `plots/`
