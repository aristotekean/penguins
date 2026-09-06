#!/usr/bin/env python
"""Entrena y publica una versión nueva de modelos en el volumen compartido.

Uso dentro del contenedor de Jupyter:

    uv run python scripts/train.py
    uv run python scripts/train.py --notas "solo RF, 400 arboles" --algoritmos randomforest
"""

from __future__ import annotations

import argparse

from penguins_ml.registry import models_dir
from penguins_ml.training import ALGORITMOS, train_and_register


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algoritmos", nargs="+", choices=ALGORITMOS, default=list(ALGORITMOS))
    parser.add_argument("--notas", default="", help="Descripción corta de la versión")
    parser.add_argument("--autor", default=None)
    parser.add_argument("--data", default=None, help="Ruta al CSV (por defecto DATA_PATH)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    print(f"Registro de modelos: {models_dir()}\n")
    train_and_register(
        args.algoritmos,
        notas=args.notas,
        autor=args.autor,
        path=args.data,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
