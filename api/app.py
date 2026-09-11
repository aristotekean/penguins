"""API de inferencia: expone los modelos publicados en el volumen compartido,
ya sea desde el contenedor de Jupyter o desde el DAG `penguins_pipeline` de
Airflow. Sin frontend propio: todo se opera desde Swagger.

La API no depende de quién entrena. Depende del contrato del registry
(`penguins_ml.registry`): una carpeta `modelo_vN` con `metadata.json` y los
`.pkl` de cada algoritmo. Cualquier productor que respete ese contrato queda
disponible para inferencia sin cambios en este código.

Dos ideas sostienen el requisito del taller —que el equipo de pruebas pueda
elegir entre las versiones que existan en cada momento—:

1. La API no carga nada en el arranque. Lee el volumen en cada request y cachea
   los .pkl invalidando por mtime. Una versión nueva queda disponible sin reiniciar.

2. El esquema OpenAPI se regenera en cada llamada a /openapi.json inyectando la
   lista real de versiones como `enum` de los parámetros `version` y `modelo`.
   En Swagger eso se renderiza como un desplegable: si desarrollo publica
   modelo_v3, al refrescar /docs el desplegable pasa a ofrecer v1, v2 y v3.

   (El enum va en parámetros de query y no en el body a propósito: Swagger UI
   edita el body como JSON crudo y ahí no dibuja ningún selector.)
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.openapi.utils import get_openapi
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from penguins_ml import registry

DESCRIPCION = """
Clasificador de especies de pingüinos del archipiélago Palmer.

Los modelos se entrenan en el contenedor de Jupyter o en el DAG `penguins_pipeline`
de Airflow, que carga el CSV en PostgreSQL, lo preprocesa y entrena desde la tabla
procesada. En ambos casos se publican en el volumen compartido y esta API los
descubre sola. El campo `autor` de cada versión indica quién la entrenó.

**Cómo elegir un modelo**

1. `GET /modelos` lista el historial completo de versiones.
2. En `POST /predict`, el desplegable **version** ofrece todas las versiones
   disponibles en este momento, y **modelo** el algoritmo dentro de ellas.
3. Si aparece una versión nueva mientras tenés esta página abierta, refrescá
   `/docs` (F5) y el desplegable se actualiza. La API no se reinicia.
"""

app = FastAPI(
    title="Penguins · API de inferencia",
    version="2.0.0",
    description=DESCRIPCION,
    swagger_ui_parameters={"tryItOutEnabled": True, "defaultModelsExpandDepth": 0},
)


# --------------------------------------------------------------------------- #
# Esquemas
# --------------------------------------------------------------------------- #
class PenguinFeatures(BaseModel):
    """Medidas morfológicas de un pingüino."""

    bill_length_mm: float = Field(gt=0, description="Largo del pico en mm (32.1–59.6).", examples=[39.1])
    bill_depth_mm: float = Field(gt=0, description="Alto del pico en mm (13.1–21.5).", examples=[18.7])
    flipper_length_mm: float = Field(gt=0, description="Largo de la aleta en mm (172–231).", examples=[181])
    body_mass_g: float = Field(gt=0, description="Masa corporal en gramos (2700–6300).", examples=[3750])
    island: Literal["Torgersen", "Biscoe", "Dream"] = Field(examples=["Torgersen"])
    sex: Literal["MALE", "FEMALE"] = Field(examples=["MALE"])

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
    version: str
    modelo: str
    species: str
    probabilities: dict[str, float]
    entrenado: str | None = None


class AlgoritmoResumen(BaseModel):
    nombre: str
    accuracy: float | None = None
    f1_macro: float | None = None


class VersionResumen(BaseModel):
    version: str
    numero: int | None = None
    creado: str | None = None
    autor: str | None = None
    notas: str = ""
    es_ultima: bool = False
    algoritmos: list[AlgoritmoResumen] = []


class ListaVersiones(BaseModel):
    total: int
    ultima: str | None = None
    disponibles: list[str] = []
    versiones: list[VersionResumen]


def _resumen(meta: dict[str, Any]) -> VersionResumen:
    modelos = meta.get("modelos", {})
    return VersionResumen(
        version=meta["version"],
        numero=meta.get("numero"),
        creado=meta.get("creado"),
        autor=meta.get("autor"),
        notas=meta.get("notas", "") or "",
        es_ultima=bool(meta.get("es_ultima")),
        algoritmos=[
            AlgoritmoResumen(
                nombre=nombre,
                accuracy=datos.get("accuracy"),
                f1_macro=datos.get("f1_macro"),
            )
            for nombre, datos in sorted(modelos.items())
        ],
    )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def raiz() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["servicio"], summary="Estado del servicio y del volumen")
def health() -> dict[str, Any]:
    versiones = registry.list_versions()
    carpeta = registry.models_dir()
    return {
        "status": "ok",
        "models_dir": str(carpeta),
        "volumen_montado": carpeta.is_dir(),
        "versiones": len(versiones),
        "ultima": versiones[0]["version"] if versiones else None,
    }


@app.get(
    "/modelos",
    response_model=ListaVersiones,
    tags=["modelos"],
    summary="Modelos disponibles",
)
def listar_modelos() -> ListaVersiones:
    """Relee el volumen y devuelve el historial de versiones, de la más nueva a
    la más vieja. Sólo aparecen las versiones que ya escribieron su `metadata.json`."""
    versiones = registry.list_versions()
    return ListaVersiones(
        total=len(versiones),
        ultima=versiones[0]["version"] if versiones else None,
        disponibles=[v["version"] for v in versiones],
        versiones=[_resumen(m) for m in versiones],
    )


@app.get("/modelos/{version}", tags=["modelos"], summary="Detalle de una versión")
def detalle_modelo(
    version: str = Path(description="Versión a inspeccionar."),
) -> dict[str, Any]:
    """Devuelve el `metadata.json` completo: métricas, dataset y entorno con el que
    se entrenó. Sirve para justificar por qué una versión se comporta distinto a otra."""
    try:
        return registry.get_metadata(version)
    except registry.RegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post(
    "/predict",
    response_model=Prediction,
    tags=["inferencia"],
    summary="Ejecutar el modelo seleccionado",
)
def predict(
    features: PenguinFeatures,
    version: str = Query(
        default="latest",
        description=(
            "Versión del modelo a ejecutar. El desplegable se arma leyendo el volumen; "
            "`latest` usa siempre la más reciente."
        ),
    ),
    modelo: str = Query(
        default="auto",
        description=(
            "Algoritmo dentro de esa versión. `auto` deja que la API elija uno de "
            "los que esa versión tenga."
        ),
    ),
) -> Prediction:
    algoritmo = None if modelo in ("auto", "") else modelo

    try:
        estimador, info = registry.load_model(version, algoritmo)
    except registry.RegistryError as exc:
        raise HTTPException(status_code=400 if algoritmo else 404, detail=str(exc)) from exc

    X = pd.DataFrame([features.model_dump()])

    try:
        species = estimador.predict(X)[0]
        probabilidades = {
            str(clase): float(p)
            for clase, p in zip(estimador.classes_, estimador.predict_proba(X)[0])
        }
    except Exception as exc:  # modelo corrupto o incompatible
        raise HTTPException(
            status_code=500,
            detail=(
                f"No se pudo ejecutar {info['algoritmo']} de {info['version']}: {exc}. "
                "Revisá que el modelo se haya entrenado con la misma versión de scikit-learn."
            ),
        ) from exc

    return Prediction(
        version=info["version"],
        modelo=info["algoritmo"],
        species=str(species),
        probabilities=probabilidades,
        entrenado=info.get("creado"),
    )


@app.post(
    "/modelos/refresh",
    tags=["servicio"],
    summary="Vaciar la caché de modelos en memoria",
)
def refresh() -> dict[str, Any]:
    """No hace falta para ver versiones nuevas: la API ya las detecta sola. Sirve
    si un `.pkl` se reemplazó en el disco conservando su fecha de modificación."""
    registry.clear_cache()
    return {"cache": "vacía", "versiones": len(registry.list_versions())}


# --------------------------------------------------------------------------- #
# OpenAPI dinámico: los desplegables de Swagger salen del volumen
# --------------------------------------------------------------------------- #
def _inyectar_enum(parametro: dict[str, Any], opciones: list[str], defecto: str) -> None:
    esquema = parametro.setdefault("schema", {})
    esquema.pop("anyOf", None)
    esquema["type"] = "string"
    esquema["enum"] = opciones
    esquema["default"] = defecto if defecto in opciones else opciones[0]


def openapi_dinamico() -> dict[str, Any]:
    """Regenera el esquema en cada request y le mete la lista real de versiones.

    A diferencia del comportamiento por defecto de FastAPI, el resultado no se
    cachea: es justamente lo que permite que el desplegable crezca cuando el
    equipo de desarrollo publica un modelo nuevo.
    """
    esquema = get_openapi(
        title=app.title,
        version=app.version,
        description=DESCRIPCION,
        routes=app.routes,
    )

    versiones = registry.list_versions()
    nombres = [v["version"] for v in versiones]
    algoritmos = sorted({a for v in versiones for a in v.get("algoritmos", [])})

    if nombres:
        detalle = ", ".join(
            f"{v['version']} ({', '.join(v['algoritmos'])})" for v in versiones
        )
        nota = f"\n\n**Versiones en el volumen ahora mismo:** {detalle}."
    else:
        nota = (
            "\n\n**El volumen todavía está vacío.** Entrene la primera versión ejecutando "
            "el DAG `penguins_pipeline` en Airflow o con "
            "`docker compose exec jupyter python scripts/train.py`, y refresque esta página."
        )
    esquema["info"]["description"] = esquema["info"].get("description", "") + nota

    for ruta in esquema.get("paths", {}).values():
        for operacion in ruta.values():
            if not isinstance(operacion, dict):
                continue
            for parametro in operacion.get("parameters", []):
                nombre = parametro.get("name")
                if nombre == "version" and nombres:
                    opciones = nombres if parametro.get("in") == "path" else ["latest"] + nombres
                    _inyectar_enum(parametro, opciones, "latest")
                elif nombre == "modelo" and algoritmos:
                    _inyectar_enum(parametro, ["auto"] + algoritmos, "auto")

    return esquema


app.openapi = openapi_dinamico  # type: ignore[method-assign]
