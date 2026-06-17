# Tesis de Maestría — Predicción de Default con LLMs y Machine Learning

[![Python](https://img.shields.io/badge/Python-3.14+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9?logo=python&logoColor=white)](https://github.com/astral-sh/uv)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-EB5E28)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-2E8B57)](https://lightgbm.readthedocs.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![Prefect](https://img.shields.io/badge/Prefect-3.6-024DFD?logo=prefect&logoColor=white)](https://www.prefect.io/)
[![LaTeX](https://img.shields.io/badge/LaTeX-MDPI%20%2F%20JRFM-008080?logo=latex&logoColor=white)](https://www.mdpi.com/journal/jrfm)
[![Status](https://img.shields.io/badge/Status-En%20revisión-yellow)](#)
[![License](https://img.shields.io/badge/License-Académica-lightgrey)](#)

Proyecto de tesis: predicción de incumplimiento crediticio en una cooperativa
del sector solidario colombiano, comparando modelos de Machine Learning
(XGBoost, LightGBM, HistGradientBoosting, Random Forest) contra un Large
Language Model (GPT-4o-mini) en configuración *Informed GPT* siguiendo la
metodología de Babaei & Giudici (2024).

**Autor:** Javier Andrés Ferro Pérez
**Asesores:** M. Andrea Arias-Serna, Jhon J. Quiza-Montealegre
**Universidad de Medellín** — Facultad de Ciencias Básicas

---

## Tabla de contenidos

1. [Requisitos previos](#1-requisitos-previos)
2. [Instalación paso a paso](#2-instalación-paso-a-paso)
3. [Cómo correr el proyecto](#3-cómo-correr-el-proyecto)
4. [Estructura del repositorio](#4-estructura-del-repositorio)
5. [Documentos de la tesis](#5-documentos-de-la-tesis)
6. [Solución de problemas](#6-solución-de-problemas)

---

## 1. Requisitos previos

| Requisito | Versión | Para qué sirve |
|---|---|---|
| Python | 3.14+ | Lenguaje del proyecto |
| `uv` | última | Gestor de dependencias y entornos virtuales |
| API Key de OpenAI | — | Necesaria solo para correr la parte de LLM (GPT-4o-mini) |
| `tectonic` (opcional) | última | Compilar el paper LaTeX a PDF |

### Instalar `uv`

`uv` es el gestor de paquetes que reemplaza a `pip` y `venv`. Instala todas las dependencias automáticamente.

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verificar la instalación:
```bash
uv --version
```

### Obtener una API Key de OpenAI

1. Ir a https://platform.openai.com/api-keys
2. Crear una nueva clave (formato: `sk-proj-...`)
3. Cargar saldo en la cuenta (correr el experimento completo cuesta aproximadamente **USD 5–10** con `gpt-4o-mini`)

> **Nota:** Si solo se quiere reproducir los resultados sin volver a llamar a la API, no hace falta la API Key. Los resultados ya están cacheados en `llm_tesis/results/` y `models/`.

---

## 2. Instalación paso a paso

### Paso 1. Clonar el repositorio

```bash
git clone https://github.com/grupoMOAI/tesis-ferro.git
cd tesis-ferro
git checkout v2
```

### Paso 2. Crear el archivo `.env`

Copiar el archivo de ejemplo y reemplazar el valor con tu clave real:

```bash
cp .env.example .env
```

Luego editar `.env` y poner tu API Key de OpenAI:

```
OPENAI_API_KEY=sk-proj-...tu_clave_aquí...
```

> El archivo `.env` está en `.gitignore`, por lo que nunca se sube al repositorio. El archivo `.env.example` sí se versiona como plantilla.

### Paso 3. Instalar dependencias

```bash
uv sync
```

`uv` crea automáticamente el entorno virtual `.venv/` e instala todas las
librerías listadas en `pyproject.toml` y `uv.lock`. Esto puede tardar 1–2
minutos la primera vez.

### Paso 4. Verificar la instalación

```bash
uv run python -c "import sklearn, xgboost, lightgbm, openai, prefect; print('OK')"
```

Si imprime `OK`, todo está bien.

---

## 3. Cómo correr el proyecto

El proyecto tiene **tres scripts principales**, cada uno con un propósito
distinto. Se recomienda correrlos en este orden la primera vez.

### 3.1 `run.py` — Reproducir resultados (recomendado para asesores)

Este es el script principal. Carga los modelos ML ya entrenados y los
resultados del LLM ya cacheados, y genera todas las gráficas y tablas del
paper.

```bash
uv run python run.py
```

**Qué hace:**
1. Carga el dataset `base/WhatsApp Business Data (1).parquet`
2. Hace el split 95% train / 5% test (seed = 1)
3. Carga los modelos ML pre-entrenados de `models/propios/`
4. Lee los resultados cacheados del LLM de `llm_tesis/results/`
5. Genera el reporte comparativo en la UI de Prefect (ver Sección 3.4)

**Tiempo estimado:** ~30 segundos.

**Si se quiere volver a llamar a la API de OpenAI** (cuesta dinero), editar
las últimas líneas de `run.py`:

```python
run_flow(
    model_source="propios",
    run_zero_shot=True,             # ← activar zero-shot
    zero_shot_skip_predict=False,   # ← False = llama a la API; True = usa cache
    run_informed_gpt=True,
    informed_gpt_n_runs=5,
    informed_gpt_example_sizes=[10, 20, 40, 80],
)
```

### 3.2 `train.py` — Reentrenar los modelos ML desde cero

Solo necesario si se quiere reproducir el entrenamiento completo de los
8 modelos ML (XGBoost, LightGBM, HistGradientBoosting, Random Forest, etc.).

```bash
uv run python train.py
```

**Qué hace:**
1. Carga el dataset y hace el split
2. Entrena cada modelo con `RandomizedSearchCV` o `GridSearchCV`
3. Guarda los `.joblib` y los `_params.json` en `models/propios/`
4. Genera el reporte comparativo

**Tiempo estimado:** 15–30 minutos según la máquina (los Random/Grid Search
son lo más pesado).

### 3.3 `regenerate_plots.py` y `generate_tables.py` — Regenerar figuras y tablas

Si se modifican los datos o modelos y hay que actualizar las imágenes y
tablas que aparecen en el paper:

```bash
uv run python regenerate_plots.py     # genera matrices de confusión y curvas ROC
uv run python generate_tables.py      # genera tablas LaTeX (Tabla 1 y 2 estilo Babaei)
```

Las salidas quedan en `plots/` y luego se copian a `document/paper/figuras/`.

### 3.4 Visualizar resultados con Prefect (opcional)

`run.py` y `train.py` están orquestados con [Prefect](https://www.prefect.io/),
que permite visualizar el flujo de ejecución y los artefactos en una UI web.

**Para activar la UI:**

```bash
# En una terminal aparte:
uv run prefect server start
```

Luego abrir http://127.0.0.1:4200 en el navegador. Allí se ven:

- Flujos ejecutados con tiempos por tarea
- Reportes en Markdown con matrices de confusión y reportes de clasificación
- Tablas comparativas entre todos los modelos

> Si la UI no está activa, los scripts igual corren — solo no se podrá ver
> el dashboard.

---

## 4. Estructura del repositorio

```
tesis_mestria_v2/
├── README.md                       ← este archivo
├── pyproject.toml                  ← dependencias y configuración del proyecto
├── uv.lock                         ← lockfile de dependencias (no editar)
├── .env                            ← API key de OpenAI (crear manualmente)
│
├── run.py                          ← Script principal: corre el experimento
├── train.py                        ← Script para reentrenar modelos ML
├── regenerate_plots.py             ← Regenera figuras del paper en inglés
├── generate_tables.py              ← Genera tablas LaTeX (Tabla 1 y 2)
│
├── base/                           ← Datos crudos
│   └── WhatsApp Business Data (1).parquet   ← Dataset (12,861 obs × 23 vars)
│
├── src/                            ← Código fuente del pipeline ML
│   ├── data.py                     ← Carga y split del dataset (95/5, seed=1)
│   ├── ml_pipeline.py              ← Orquestador del entrenamiento de los 8 modelos
│   ├── evaluation.py               ← Métricas, matrices de confusión, ROC
│   ├── persistence.py              ← Guardar/cargar .joblib + _params.json
│   ├── summary.py                  ← Reporte comparativo final
│   └── models/                     ← Cada archivo entrena un modelo
│       ├── xgboost_tuned.py
│       ├── lgbm_tuned.py
│       ├── lgbm_skrub.py
│       ├── hist_gradient.py
│       ├── random_forest.py
│       ├── logistic_regression.py
│       ├── logistic_regression_balanced.py
│       └── mlp.py
│
├── llm_tesis/                      ← Código del experimento LLM
│   ├── client.py                   ← Cliente OpenAI con retry/backoff
│   ├── prompt.py                   ← Plantillas de prompts (sistema y usuario)
│   ├── predict.py                  ← Predicción row-by-row con caché en CSV
│   ├── evaluate.py                 ← Métricas y plots para el LLM
│   ├── zero_shot/flow.py           ← Flujo Zero-shot (sin ejemplos)
│   ├── informed_gpt/flow.py        ← Flujo Informed GPT (con N ejemplos)
│   └── results/                    ← Resultados cacheados (CSV)
│       ├── llm_predictions.csv         ← Predicciones zero-shot
│       ├── experiment_preds.csv        ← Todas las predicciones de Informed GPT
│       └── experiment_results.csv      ← AUC por config + run
│
├── prompts/                        ← Prompts del sistema en texto plano
│   ├── system_zero_shot_n0.txt
│   ├── system_informed_gpt_n10.txt
│   ├── system_informed_gpt_n20.txt
│   ├── system_informed_gpt_n40.txt
│   ├── system_informed_gpt_n80.txt
│   └── user_ejemplo.txt
│
├── models/                         ← Modelos ML pre-entrenados
│   ├── propios/                    ← Modelos entrenados con `train.py`
│   │   ├── xgb_model.joblib
│   │   ├── lgb_model.joblib
│   │   ├── lgbm_skrub.joblib
│   │   ├── hist_model.joblib
│   │   ├── rfc_model.joblib
│   │   └── *_params.json           ← Hiperparámetros optimizados
│   └── jhquiza/                    ← Modelos del asesor (referencia)
│
├── plots/                          ← Figuras generadas (PNG)
│   ├── comparacion_roc_llm_vs_ml.png
│   ├── *_confusion_matrix.png
│   └── tabla*.png
│
├── document/                       ← Documentos LaTeX de la tesis
│   ├── paper/                      ← Paper para JRFM (formato MDPI)
│   │   ├── paper.tex
│   │   ├── paper.pdf               ← ← ← VERSIÓN COMPILADA DEL PAPER
│   │   ├── referencias.bib
│   │   ├── figuras/                ← Imágenes embebidas en el paper
│   │   ├── tablas/                 ← Tablas LaTeX (auto-generadas)
│   │   ├── Definitions/            ← Clase MDPI (mdpi.cls + logos)
│   │   ├── template.tex            ← Template oficial JRFM (referencia)
│   │   └── examples/               ← Papers JRFM publicados (referencia de forma)
│   └── review/                     ← Instrucciones para autores JRFM
│
├── docs/                           ← Papers fuente clave
│   ├── GPT classifications, with application to credit lending.pdf   (Babaei & Giudici 2024)
│   └── Machine Learning Model for default risk prediction in Colombia's solidarity sector.pdf  (Arias-Serna 2025)
│
└── .claude/                        ← Configuración del asistente Claude Code (no necesario)
```

---

## 5. Documentos de la tesis

El paper en formato JRFM (Journal of Risk and Financial Management) está en:

**`document/paper/paper.pdf`**

Para recompilar el PDF (opcional, requiere `tectonic`):

```bash
brew install tectonic                         # macOS
# o ver https://tectonic-typesetting.github.io/ para otras plataformas

cd document/paper
tectonic paper.tex
```

Las instrucciones para autores y el template oficial de la revista están en
**`document/review/`**.

---

## 6. Solución de problemas

### `uv: command not found`
`uv` no está en el PATH. Reabrir la terminal después de instalarlo, o agregar manualmente:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### `OPENAI_API_KEY not found` o errores de autenticación
Verificar que el archivo `.env` existe en la raíz del proyecto y contiene la
clave en el formato exacto:
```
OPENAI_API_KEY=sk-proj-...
```
Sin comillas, sin espacios alrededor del `=`.

### El experimento de LLM tarda mucho / cuesta mucho
Por defecto `run.py` usa el caché en `llm_tesis/results/`, así que **no
llama a la API**. Si se quiere correr de cero, el costo aproximado con
`gpt-4o-mini` es:

- Zero-shot (200 muestras): ~USD 0.50
- Informed GPT (5 runs × 4 tamaños × 644 muestras): ~USD 8

### `ImportError: skrub._to_float32`
`regenerate_plots.py` y `generate_tables.py` cargan modelos guardados con
una versión vieja de `skrub`. El script ya incluye un *shim* automático,
así que solo hay que correrlo con `uv run python ...` (no con `python`
directo).

### El PDF del paper no compila
El error más común es `Unable to load picture or PDF file`. Verificar que
existe:
```bash
ls document/paper/Definitions/logo-mdpi.png
ls document/paper/figuras/
```
Si falta alguna figura, regenerarla con `uv run python regenerate_plots.py`.

### Prefect UI no carga en `http://127.0.0.1:4200`
Asegurarse de haber corrido `uv run prefect server start` en una terminal
**antes** de correr `run.py` o `train.py`.

---

## Contacto

Para preguntas sobre el código o los experimentos: **andresferro6@hotmail.com**
