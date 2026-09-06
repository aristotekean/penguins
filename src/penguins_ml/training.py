"""Entrenamiento de clasificadores de especie sobre el dataset Palmer Penguins.

El resultado de cada corrida se publica como una **versión nueva** en el
volumen compartido, vía `penguins_ml.registry.save_version`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import registry

NUMERIC_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_FEATURES = ["island", "sex"]
TARGET = "species"

ALGORITMOS = ("randomforest", "decisiontree", "logisticregression")

DEFAULTS: dict[str, dict[str, Any]] = {
    "randomforest": {"n_estimators": 200, "random_state": 42},
    "decisiontree": {"random_state": 42},
    "logisticregression": {"max_iter": 500},
}


def data_path() -> Path:
    """Ruta del CSV. Configurable con DATA_PATH."""
    return Path(os.getenv("DATA_PATH", "/workspace/data/penguins.csv"))


def load_data(path: Path | str | None = None) -> pd.DataFrame:
    ruta = Path(path) if path else data_path()
    df = pd.read_csv(ruta, na_values=["NA", "."])
    return df.dropna()


def build_pipeline(algoritmo: str, **params: Any) -> Pipeline:
    """Preprocesamiento + clasificador en un solo objeto serializable."""
    escalar = algoritmo == "logisticregression"
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler() if escalar else "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    opciones = {**DEFAULTS.get(algoritmo, {}), **params}
    if algoritmo == "randomforest":
        model = RandomForestClassifier(**opciones)
    elif algoritmo == "decisiontree":
        model = DecisionTreeClassifier(**opciones)
    elif algoritmo == "logisticregression":
        model = LogisticRegression(**opciones)
    else:
        raise ValueError(f"Algoritmo no soportado: {algoritmo!r}. Usá {ALGORITMOS}.")

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def train_and_register(
    algoritmos: Iterable[str] = ALGORITMOS,
    *,
    notas: str = "",
    autor: str | None = None,
    path: Path | str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    params: dict[str, dict[str, Any]] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Entrena, evalúa y publica una versión nueva. Devuelve su metadata."""
    algoritmos = list(algoritmos)
    params = params or {}

    df = load_data(path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    estimadores: dict[str, Pipeline] = {}
    metricas: dict[str, dict[str, Any]] = {}

    for algoritmo in algoritmos:
        pipeline = build_pipeline(algoritmo, **params.get(algoritmo, {}))
        pipeline.fit(X_train, y_train)

        pred = pipeline.predict(X_test)
        accuracy = float(accuracy_score(y_test, pred))
        f1 = float(f1_score(y_test, pred, average="macro"))

        estimadores[algoritmo] = pipeline
        metricas[algoritmo] = {
            "accuracy": round(accuracy, 4),
            "f1_macro": round(f1, 4),
            "params": {k: str(v) for k, v in pipeline.named_steps["model"].get_params().items()},
            "reporte": classification_report(y_test, pred, output_dict=True, zero_division=0),
        }

        if verbose:
            print(f"{algoritmo:<20} accuracy={accuracy:.4f}  f1_macro={f1:.4f}")

    meta = registry.save_version(
        estimadores,
        metricas=metricas,
        notas=notas,
        autor=autor,
        dataset={
            "archivo": str(Path(path) if path else data_path()),
            "filas": int(len(df)),
            "train": int(len(X_train)),
            "test": int(len(X_test)),
            "clases": sorted(y.unique().tolist()),
            "test_size": test_size,
            "random_state": random_state,
        },
    )

    if verbose:
        print(f"\nVersión publicada: {meta['version']} → {registry.models_dir()}")
        print("La API ya la ve, no hace falta reiniciar nada.")

    return meta
