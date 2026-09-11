"""Load data/penguins.csv into the postgres-penguins database using SQLAlchemy ORM."""

import os
from datetime import datetime

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base

PENGUINS_DB_URI = os.getenv(
    "PENGUINS_DB_URI",
    "postgresql+psycopg2://penguins:penguins@postgres-penguins:5432/penguins",
)
CSV_PATH = os.getenv("PENGUINS_CSV_PATH", "/opt/airflow/data/penguins.csv")

Base = declarative_base()


class Penguin(Base):
    __tablename__ = "penguins"

    id = Column(Integer, primary_key=True, autoincrement=True)
    species = Column(String(32), nullable=False)
    island = Column(String(32), nullable=False)
    bill_length_mm = Column(Float, nullable=True)
    bill_depth_mm = Column(Float, nullable=True)
    flipper_length_mm = Column(Float, nullable=True)
    body_mass_g = Column(Float, nullable=True)
    sex = Column(String(8), nullable=True)


def get_engine():
    return create_engine(PENGUINS_DB_URI)


def create_table():
    Base.metadata.create_all(get_engine())


def load_csv():
    df = pd.read_csv(CSV_PATH, na_values=["NA", "."])
    # NaN is not a valid SQL value; convert it to None so the ORM writes NULL.
    df = df.astype(object).where(pd.notnull(df), None)

    rows = [Penguin(**record) for record in df.to_dict(orient="records")]

    with Session(get_engine()) as session, session.begin():
        # Idempotent load: wipe previous contents before inserting.
        session.query(Penguin).delete()
        session.add_all(rows)

    print(f"Loaded {len(rows)} rows into '{Penguin.__tablename__}'")


with DAG(
    dag_id="load_dataset_db",
    description="Load the penguins CSV into the postgres-penguins database",
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["penguins", "etl"],
) as dag:
    create_table_task = PythonOperator(
        task_id="create_table",
        python_callable=create_table,
    )

    load_csv_task = PythonOperator(
        task_id="load_csv",
        python_callable=load_csv,
    )

    create_table_task >> load_csv_task
