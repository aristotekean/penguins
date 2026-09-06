"""Registro de modelos versionados sobre el volumen compartido.

Este módulo es la pieza que comparten los dos contenedores:

- Jupyter lo usa para **escribir** una versión nueva (`save_version`).
- FastAPI lo usa para **descubrir y leer** las versiones disponibles
  (`list_versions`, `load_model`).

Layout en disco (dentro del volumen montado en MODELS_DIR):

    /models/
    ├── modelo_v1/
    │   ├── metadata.json          <- marca de "versión completa"
    │   ├── randomforest.pkl
    │   ├── decisiontree.pkl
    │   └── logisticregression.pkl
    ├── modelo_v2/
    └── modelo_v3/

Reglas de concurrencia
----------------------
1. El número de versión se reserva con `mkdir`, que es atómico: si dos
   entrenamientos corren a la vez, uno de los dos falla el mkdir y reintenta
   con el número siguiente. Nunca se sobrescribe una versión existente.
2. `metadata.json` se escribe al final, en un temporal + `rename` (atómico en
   el mismo filesystem). La API sólo lista carpetas que ya tienen
   `metadata.json`, así que jamás ve una versión a medio escribir.
"""

from __future__ import annotations

import json
import os
import platform
import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

VERSION_PATTERN = re.compile(r"^modelo_v(\d+)$")
METADATA_FILE = "metadata.json"
LATEST = "latest"

_CACHE_MAX = 8
_cache: "OrderedDict[tuple[str, float], Any]" = OrderedDict()


class RegistryError(Exception):
    """Error de dominio del registro (se traduce a 404/400 en la API)."""


def models_dir() -> Path:
    """Carpeta raíz del registro. Se configura con la variable MODELS_DIR."""
    return Path(os.getenv("MODELS_DIR", "/models"))


# --------------------------------------------------------------------------- #
# Lectura
# --------------------------------------------------------------------------- #
def version_number(nombre: str) -> int | None:
    match = VERSION_PATTERN.match(nombre)
    return int(match.group(1)) if match else None


def _read_metadata(carpeta: Path) -> dict[str, Any] | None:
    archivo = carpeta / METADATA_FILE
    if not archivo.is_file():
        return None  # versión incompleta o interrumpida: se ignora
    try:
        meta = json.loads(archivo.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(meta, dict):
        return None

    meta.setdefault("version", carpeta.name)
    meta.setdefault("numero", version_number(carpeta.name))
    modelos = meta.get("modelos")
    if not isinstance(modelos, dict) or not modelos:
        # Fallback: si el metadata quedó corto, se descubren los .pkl del disco.
        modelos = {p.stem: {"archivo": p.name} for p in sorted(carpeta.glob("*.pkl"))}
        meta["modelos"] = modelos
    meta["algoritmos"] = sorted(modelos)
    return meta


def list_versions() -> list[dict[str, Any]]:
    """Todas las versiones completas del volumen, de la más nueva a la más vieja."""
    base = models_dir()
    if not base.is_dir():
        return []

    encontradas: list[dict[str, Any]] = []
    for carpeta in base.iterdir():
        if not carpeta.is_dir() or version_number(carpeta.name) is None:
            continue
        meta = _read_metadata(carpeta)
        if meta is not None:
            encontradas.append(meta)

    encontradas.sort(key=lambda m: m.get("numero") or 0, reverse=True)
    for i, meta in enumerate(encontradas):
        meta["es_ultima"] = i == 0
    return encontradas


def latest_version() -> str | None:
    versiones = list_versions()
    return versiones[0]["version"] if versiones else None


def resolve_version(version: str | None) -> str:
    """Normaliza `None`/`latest`/`modelo_vN` a un nombre de carpeta real."""
    pedido = (version or LATEST).strip()

    if pedido == LATEST:
        ultima = latest_version()
        if ultima is None:
            raise RegistryError(
                "Todavía no hay modelos en el volumen. Entrená uno desde Jupyter."
            )
        return ultima

    if version_number(pedido) is None:
        raise RegistryError(
            f"Nombre de versión inválido: {pedido!r}. Usá 'latest' o 'modelo_vN'."
        )

    if not (models_dir() / pedido / METADATA_FILE).is_file():
        disponibles = [v["version"] for v in list_versions()]
        raise RegistryError(
            f"La versión {pedido!r} no existe. Disponibles: {disponibles or 'ninguna'}."
        )
    return pedido


def get_metadata(version: str | None = None) -> dict[str, Any]:
    resuelta = resolve_version(version)
    meta = _read_metadata(models_dir() / resuelta)
    if meta is None:
        raise RegistryError(f"No se pudo leer el metadata de {resuelta!r}.")
    meta["es_ultima"] = resuelta == latest_version()
    return meta


def _load_cached(ruta: Path) -> Any:
    """Carga con caché invalidada por mtime: un .pkl nuevo se recarga solo."""
    mtime = ruta.stat().st_mtime
    clave = (str(ruta), mtime)
    if clave in _cache:
        _cache.move_to_end(clave)
        return _cache[clave]

    objeto = joblib.load(ruta)
    _cache[clave] = objeto
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)
    return objeto


def load_model(
    version: str | None = None, algoritmo: str | None = None
) -> tuple[Any, dict[str, Any]]:
    """Devuelve `(estimador, info)` donde info trae version/algoritmo resueltos."""
    meta = get_metadata(version)
    disponibles: list[str] = meta["algoritmos"]

    elegido = algoritmo or (disponibles[0] if len(disponibles) == 1 else None)
    if elegido is None:
        preferidos = [a for a in ("randomforest", "decisiontree") if a in disponibles]
        elegido = preferidos[0] if preferidos else disponibles[0]

    if elegido not in disponibles:
        raise RegistryError(
            f"El algoritmo {elegido!r} no existe en {meta['version']}. "
            f"Disponibles: {disponibles}."
        )

    archivo = meta["modelos"][elegido].get("archivo", f"{elegido}.pkl")
    ruta = models_dir() / meta["version"] / archivo
    if not ruta.is_file():
        raise RegistryError(f"Falta el archivo {archivo!r} en {meta['version']}.")

    info = {
        "version": meta["version"],
        "algoritmo": elegido,
        "creado": meta.get("creado"),
        "entorno": meta.get("entorno", {}),
    }
    return _load_cached(ruta), info


def clear_cache() -> None:
    _cache.clear()


# --------------------------------------------------------------------------- #
# Escritura
# --------------------------------------------------------------------------- #
def _reserve_version_dir() -> tuple[int, Path]:
    """Reserva `modelo_vN` con mkdir (atómico). Reintenta si hay carrera."""
    base = models_dir()
    base.mkdir(parents=True, exist_ok=True)

    for _ in range(100):
        usados = [
            n
            for carpeta in base.iterdir()
            if carpeta.is_dir() and (n := version_number(carpeta.name)) is not None
        ]
        numero = max(usados, default=0) + 1
        carpeta = base / f"modelo_v{numero}"
        try:
            carpeta.mkdir()
        except FileExistsError:
            continue
        return numero, carpeta

    raise RegistryError("No se pudo reservar un número de versión.")


def _entorno() -> dict[str, str]:
    import sklearn

    datos = {"python": platform.python_version(), "scikit_learn": sklearn.__version__}
    for nombre, modulo in (("pandas", "pandas"), ("joblib", "joblib"), ("numpy", "numpy")):
        try:
            datos[nombre] = __import__(modulo).__version__
        except Exception:  # pragma: no cover
            pass
    return datos


def save_version(
    estimadores: dict[str, Any],
    *,
    metricas: dict[str, dict[str, Any]] | None = None,
    notas: str = "",
    autor: str | None = None,
    dataset: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Guarda un entrenamiento como una versión nueva, sin tocar las anteriores.

    Devuelve el metadata escrito.
    """
    if not estimadores:
        raise RegistryError("No hay estimadores para guardar.")

    metricas = metricas or {}
    numero, carpeta = _reserve_version_dir()

    modelos_meta: dict[str, Any] = {}
    for algoritmo, estimador in estimadores.items():
        archivo = f"{algoritmo}.pkl"
        joblib.dump(estimador, carpeta / archivo)
        modelos_meta[algoritmo] = {"archivo": archivo, **metricas.get(algoritmo, {})}

    meta: dict[str, Any] = {
        "version": carpeta.name,
        "numero": numero,
        "creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "autor": autor or os.getenv("AUTOR", "equipo-desarrollo"),
        "notas": notas,
        "dataset": dataset or {},
        "entorno": _entorno(),
        "modelos": modelos_meta,
    }
    if extra:
        meta.update(extra)

    # metadata.json es la marca de "versión lista": se publica de forma atómica.
    temporal = carpeta / f".{METADATA_FILE}.tmp"
    temporal.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    os.replace(temporal, carpeta / METADATA_FILE)

    meta["algoritmos"] = sorted(modelos_meta)
    return meta
