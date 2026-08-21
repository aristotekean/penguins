"""Train a species classifier on the Palmer Penguins dataset."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).parent / "penguins.csv"
MODEL_PATH = Path(__file__).parent / "penguin_model.pkl"

NUMERIC_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_FEATURES = ["island", "sex"]
TARGET = "species"

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["NA", "."])
    return df.dropna()


def build_pipeline(model_type: str) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    if model_type == "randomforest":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    elif model_type == "decisiontree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == "logisticregression":
        model = LogisticRegression(max_iter=500)
    else:
        raise ValueError("Modelo no soportado")
    
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    df = load_data(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    modelos = ["randomforest", "decisiontree", "logisticregression"]

    for modelo in modelos:
        print(f"\nEntrenando modelo: {modelo}")

        pipeline = build_pipeline(modelo)
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        print(f"Accuracy {modelo}: {accuracy_score(y_test, predictions):.4f}")
        print(classification_report(y_test, predictions))

        output_path = Path(f"model_{modelo}.pkl")
        joblib.dump(pipeline, output_path)
        print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
