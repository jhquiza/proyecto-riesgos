# add-model

Adds a new ML model to the default prediction pipeline following the project's existing pattern.

## Instructions

When the user asks to add a new classifier, follow these steps:

### 1. Create the model file at `src/models/<nombre_modelo>.py`

Follow the exact pattern used by existing models. Example structure:

```python
from prefect import task
from sklearn.pipeline import Pipeline
from skrub import TableVectorizer
# Import the classifier
from sklearn.xxx import XxxClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
import numpy as np


@task(name="entrenar_<nombre_modelo>", log_prints=True)
def train(X_train, y_train):
    pipeline = Pipeline([
        ("preprocessor", TableVectorizer()),
        ("classifier", XxxClassifier(
            class_weight="balanced",
            random_state=1,
        )),
    ])

    param_grid = {
        # Define search space here
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=20,
        scoring="f1_weighted",
        cv=StratifiedKFold(n_splits=5),
        random_state=1,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    print(f"Mejor score CV: {search.best_score_:.4f}")
    print(f"Mejores params: {search.best_params_}")
    return search
```

### 2. Register the model in `main.py`

Add an import at the top:
```python
from src.models import <nombre_modelo>
```

Add to the `ML_MODELS` list:
```python
("<artifact_name>", <nombre_modelo>, "best_estimator_"),
```
Use `None` as the third element (instead of `"best_estimator_"`) only if the model is not wrapped in a search (like `lgbm_skrub`).

### 3. Conventions to follow

- All comments and print statements in **Spanish**
- Use `@task(name="entrenar_<nombre>", log_prints=True)` decorator
- Use `TableVectorizer()` as preprocessor in sklearn `Pipeline`
- Use `class_weight="balanced"` or equivalent for imbalanced classes
- Use `random_state=1` for reproducibility
- Scoring metric: `f1_weighted`
- Cross-validation: `StratifiedKFold(n_splits=5)`

### 4. Run and verify

```bash
uv run python main.py
```

Check the Prefect UI at http://127.0.0.1:4200 to verify the model trained and artifacts were created.
