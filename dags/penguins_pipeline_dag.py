"""Penguins ML pipeline: wipe DB -> load raw CSV -> preprocess -> train.

Tables (postgres-penguins):
  penguins_raw        exact copy of data/penguins.csv, every column as text
  penguins_processed  cleaned, typed rows with a train/test split label

The trained pipelines are published as a new version in MODELS_DIR through
penguins_ml.registry, so the FastAPI service can serve them immediately.
"""

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base

# Airflow's task runner imports dill, which patches pickle._Pickler.dispatch.
# joblib's pickler copies that table when it is first imported, so the reset
# must happen here, before any sklearn/joblib import. Otherwise slice objects
# inside ColumnTransformer are pickled as dill references and the API, which
# has no dill installed, cannot load the model.
try:
    import dill

    dill.extend(use_dill=False)
except ImportError:
    pass

PENGUINS_DB_URI = os.getenv(
    "PENGUINS_DB_URI",
    "postgresql+psycopg2://penguins:penguins@postgres-penguins:5432/penguins",
)
CSV_PATH = os.getenv("PENGUINS_CSV_PATH", "/opt/airflow/data/penguins.csv")

NUMERIC_COLUMNS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_COLUMNS = ["island", "sex"]
TARGET = "species"
MISSING_TOKENS = {"NA", ".", ""}

TEST_SIZE = 0.2
RANDOM_STATE = 42

Base = declarative_base()


class PenguinRaw(Base):
    """Row of the CSV as-is. Everything is text so nothing is interpreted."""

    __tablename__ = "penguins_raw"

    id = Column(Integer, primary_key=True, autoincrement=True)
    species = Column(String(32))
    island = Column(String(32))
    bill_length_mm = Column(String(16))
    bill_depth_mm = Column(String(16))
    flipper_length_mm = Column(String(16))
    body_mass_g = Column(String(16))
    sex = Column(String(16))


class PenguinProcessed(Base):
    """Cleaned, typed row ready for training, tagged with its split."""

    __tablename__ = "penguins_processed"

    id = Column(Integer, primary_key=True, autoincrement=True)
    species = Column(String(32), nullable=False)
    island = Column(String(32), nullable=False)
    bill_length_mm = Column(Float, nullable=False)
    bill_depth_mm = Column(Float, nullable=False)
    flipper_length_mm = Column(Float, nullable=False)
    body_mass_g = Column(Float, nullable=False)
    sex = Column(String(8), nullable=False)
    split = Column(String(8), nullable=False)  # "train" | "test"


def get_engine():
    return create_engine(PENGUINS_DB_URI)


def read_table(model, engine):
    """Load a whole ORM table into a DataFrame (without the surrogate id)."""
    import pandas as pd

    columns = [c.name for c in model.__table__.columns if c.name != "id"]
    with Session(engine) as session:
        rows = session.query(model).order_by(model.id).all()
    return pd.DataFrame([{c: getattr(r, c) for c in columns} for r in rows], columns=columns)


# --------------------------------------------------------------------------- #
# Task 1: wipe the database
# --------------------------------------------------------------------------- #
def wipe_db():
    from sqlalchemy import MetaData

    engine = get_engine()

    # Reflect whatever exists so the wipe covers every table, not only ours.
    existing = MetaData()
    existing.reflect(bind=engine)
    existing.drop_all(engine)

    Base.metadata.create_all(engine)
    print(f"Dropped: {sorted(existing.tables) or 'nothing'}")
    print(f"Created: {sorted(Base.metadata.tables)}")


# --------------------------------------------------------------------------- #
# Task 2: load the CSV without any preprocessing
# --------------------------------------------------------------------------- #
def load_raw_data():
    import pandas as pd

    # dtype=str + keep_default_na=False: "NA" and "." stay as literal text.
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    rows = [PenguinRaw(**record) for record in df.to_dict(orient="records")]

    with Session(get_engine()) as session, session.begin():
        session.add_all(rows)

    print(f"Loaded {len(rows)} raw rows into '{PenguinRaw.__tablename__}'")
    return len(rows)


# --------------------------------------------------------------------------- #
# Task 3: preprocess for training
# --------------------------------------------------------------------------- #
def preprocess_data():
    import pandas as pd
    from sklearn.model_selection import train_test_split

    engine = get_engine()
    df = read_table(PenguinRaw, engine)
    total = len(df)

    # 1. Missing-value tokens -> NULL, then numeric casting.
    df = df.replace(list(MISSING_TOKENS), pd.NA)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Normalise categorical text.
    for col in CATEGORICAL_COLUMNS + [TARGET]:
        df[col] = df[col].str.strip()
    df["sex"] = df["sex"].str.upper()

    # 3. Drop incomplete rows and exact duplicates.
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    # 4. Stratified train/test split persisted as a column, so training
    #    always evaluates on the same held-out rows.
    train_idx, test_idx = train_test_split(
        df.index, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    rows = [PenguinProcessed(**record) for record in df.to_dict(orient="records")]
    with Session(engine) as session, session.begin():
        session.query(PenguinProcessed).delete()
        session.add_all(rows)

    summary = {
        "raw_rows": total,
        "processed_rows": len(rows),
        "dropped_rows": total - len(rows),
        "train": int(len(train_idx)),
        "test": int(len(test_idx)),
    }
    print(summary)
    return summary


# --------------------------------------------------------------------------- #
# Task 4: train from the preprocessed table
# --------------------------------------------------------------------------- #
def train_model():
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    from penguins_ml import registry
    from penguins_ml.training import ALGORITMOS, build_pipeline

    df = read_table(PenguinProcessed, get_engine())
    features = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    X_train, y_train = train[features], train[TARGET]
    X_test, y_test = test[features], test[TARGET]

    estimators = {}
    metrics = {}
    for algorithm in ALGORITMOS:
        pipeline = build_pipeline(algorithm)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)

        accuracy = float(accuracy_score(y_test, pred))
        f1 = float(f1_score(y_test, pred, average="macro"))
        estimators[algorithm] = pipeline
        metrics[algorithm] = {
            "accuracy": round(accuracy, 4),
            "f1_macro": round(f1, 4),
            "params": {k: str(v) for k, v in pipeline.named_steps["model"].get_params().items()},
            "reporte": classification_report(y_test, pred, output_dict=True, zero_division=0),
        }
        print(f"{algorithm:<20} accuracy={accuracy:.4f}  f1_macro={f1:.4f}")

    meta = registry.save_version(
        estimators,
        metricas=metrics,
        notas="Trained by Airflow DAG penguins_pipeline from table penguins_processed",
        autor="airflow",
        dataset={
            "archivo": f"{PENGUINS_DB_URI.rsplit('@', 1)[-1]}/{PenguinProcessed.__tablename__}",
            "filas": int(len(df)),
            "train": int(len(train)),
            "test": int(len(test)),
            "clases": sorted(df[TARGET].unique().tolist()),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
    )
    print(f"Published {meta['version']} -> {registry.models_dir()}")
    return {"version": meta["version"], **{a: m["accuracy"] for a, m in metrics.items()}}


with DAG(
    dag_id="penguins_pipeline",
    description="Wipe DB, load raw penguins CSV, preprocess, and train species classifiers",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["penguins", "mlops"],
) as dag:
    wipe_db_task = PythonOperator(task_id="wipe_db", python_callable=wipe_db)
    load_raw_data_task = PythonOperator(task_id="load_raw_data", python_callable=load_raw_data)
    preprocess_data_task = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    train_model_task = PythonOperator(task_id="train_model", python_callable=train_model)

    wipe_db_task >> load_raw_data_task >> preprocess_data_task >> train_model_task
