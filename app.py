"""FastAPI server exposing the penguin species classifier."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "penguin_model.pkl"

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="Penguin Species Classifier", version="0.1.0", lifespan=lifespan)


class PenguinFeatures(BaseModel):
    """Morphological measurements of a penguin from the Palmer Archipelago."""

    bill_length_mm: float = Field(
        gt=0,
        description="Bill (culmen) length in millimeters. Observed range: 32.1-59.6.",
        examples=[39.1],
    )
    bill_depth_mm: float = Field(
        gt=0,
        description="Bill (culmen) depth in millimeters. Observed range: 13.1-21.5.",
        examples=[18.7],
    )
    flipper_length_mm: float = Field(
        gt=0,
        description="Flipper length in millimeters. Observed range: 172-231.",
        examples=[181],
    )
    body_mass_g: float = Field(
        gt=0,
        description="Body mass in grams. Observed range: 2700-6300.",
        examples=[3750],
    )
    island: Literal["Torgersen", "Biscoe", "Dream"] = Field(
        description="Island in the Palmer Archipelago where the penguin was observed.",
        examples=["Torgersen"],
    )
    sex: Literal["MALE", "FEMALE"] = Field(
        description="Sex of the penguin (uppercase, as recorded in the dataset).",
        examples=["MALE"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "bill_length_mm": 39.1,
                    "bill_depth_mm": 18.7,
                    "flipper_length_mm": 181,
                    "body_mass_g": 3750,
                    "island": "Torgersen",
                    "sex": "MALE",
                }
            ]
        }
    }


class Prediction(BaseModel):
    species: str
    probabilities: dict[str, float]

@app.post("/predict", response_model=Prediction, summary="Predict penguin species")
def predict(features: PenguinFeatures) -> Prediction:
    """Classify a penguin as Adelie, Chinstrap, or Gentoo from its measurements."""
    X = pd.DataFrame([features.model_dump()])
    species = model.predict(X)[0]
    probabilities = dict(zip(model.classes_, model.predict_proba(X)[0]))
    return Prediction(species=species, probabilities=probabilities)
