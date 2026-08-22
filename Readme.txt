README – PENGUINS CLASSIFIER
Documentación técnica ejecutiva – nivel MSc

1. PROPÓSITO

El proyecto implementa un sistema de clasificación de especies de pingüinos mediante Machine Learning y una API REST. La arquitectura separa las etapas de entrenamiento, inferencia y despliegue:

Datos → Entrenamiento → Modelo → API → Docker → VM → Usuario/Integración

Repositorio:
https://github.com/aristotekean/penguins

2. FLUJO DEL PROCESO

El dataset penguins.csv es procesado mediante train.py. Se utilizan cuatro variables numéricas:

- bill_length_mm
- bill_depth_mm
- flipper_length_mm
- body_mass_g

y dos variables categóricas:

- island
- sex

La variable objetivo es species.

Los datos incompletos son eliminados y las variables categóricas son transformadas mediante OneHotEncoder. El conjunto se divide en entrenamiento y prueba (80/20), manteniendo la proporción de clases.

3. MODELOS

Se implementan y comparan tres alternativas:

- Random Forest: 200 árboles.
- Decision Tree.
- Logistic Regression.

Cada alternativa se integra en un pipeline de preprocesamiento + clasificación y se evalúa mediante Accuracy y classification_report.

Los modelos entrenados se serializan como:

model_randomforest.pkl
model_decisiontree.pkl
model_logisticregression.pkl

4. API DE INFERENCIA

app.py implementa FastAPI y carga los tres modelos al iniciar el servicio.

Endpoint principal:

POST /predict

El usuario o sistema integrador selecciona el modelo y suministra las características del pingüino. La respuesta contiene:

- especie predicha
- probabilidades por especie

La API dispone además de documentación interactiva mediante /docs.

5. GESTIÓN DEL ENTORNO

El proyecto utiliza uv para gestionar y reproducir el entorno Python.

Python requerido: >= 3.12

Las dependencias se declaran en pyproject.toml y se fijan mediante uv.lock.

Comandos principales:

uv sync
uv run python train.py
uv run uvicorn app:app --host 0.0.0.0 --port 8025

6. CONTROL DE VERSIONES

El repositorio utiliza Git y GitHub como mecanismo de control y trazabilidad del desarrollo.

Rama principal actual:

main

Flujo recomendado para evolución:

feature/* → validación → Pull Request → main

Los cambios deben mantenerse en commits pequeños y descriptivos, permitiendo identificar la evolución del código, modelos y configuración.

7. .GITIGNORE

El proyecto utiliza .gitignore para excluir archivos generados, cachés, entornos virtuales, logs y archivos de configuración local.

En particular, se excluyen:

.env
.venv/
venv/
env/
__pycache__/

Esto evita incorporar al repositorio información sensible o artefactos propios del entorno de desarrollo.

Las credenciales, tokens, claves VPN y secretos no deben almacenarse en Git.

8. CONTENEDORIZACIÓN

Docker permite empaquetar la API junto con su entorno de ejecución.

El Dockerfile utiliza dos etapas:

1. Instalación reproducible de dependencias mediante uv y uv.lock.
2. Imagen runtime ligera basada en Python 3.12.

La aplicación se ejecuta mediante Uvicorn en el puerto 8025.

Construcción:

docker build -t penguins-api .

Ejecución:

docker run --rm -p 8025:8025 penguins-api

9. DESPLIEGUE EN VM

La imagen Docker se ejecuta dentro de una VM, separando el entorno de desarrollo del entorno de servicio.

Arquitectura:

Usuario / Integración
        ↓
        VM
        ↓
     Docker
        ↓
     FastAPI
        ↓
  Modelo ML

La VM debe disponer de Docker, conectividad de red y acceso al puerto definido para la API.

10. VPN Y RED

Cuando la VM pertenece a una red privada, el acceso se realiza mediante la VPN correspondiente.

La validación del entorno debe considerar:

- conectividad con la VM
- resolución DNS
- rutas de red
- firewall
- puerto 8025
- acceso al endpoint de la API

La VPN y las credenciales asociadas son elementos de infraestructura y seguridad; no deben formar parte del repositorio.

11. TRAZABILIDAD DEL DESPLIEGUE

La reproducibilidad se soporta mediante:

Git/GitHub → código y versiones
uv.lock → dependencias
PKL → modelos entrenados
Dockerfile → entorno de ejecución
VM → infraestructura de despliegue

Ante una modificación del modelo, el flujo es:

Datos/código → entrenamiento → evaluación → nuevo PKL → pruebas → nueva imagen Docker → despliegue.

12. ESTRUCTURA PRINCIPAL

penguins/
├── train.py
├── app.py
├── penguins.csv
├── model_randomforest.pkl
├── model_decisiontree.pkl
├── model_logisticregression.pkl
├── Dockerfile
├── .dockerignore
├── .gitignore
├── pyproject.toml
└── uv.lock

13. SÍNTESIS TÉCNICA

La solución implementa un flujo completo de Machine Learning orientado a servicio:

1. Preparación de datos.
2. Entrenamiento y comparación de tres modelos.
3. Serialización de los modelos.
4. Exposición mediante API REST.
5. Contenerización con Docker.
6. Despliegue en VM.
7. Consumo por usuario final o integración.

El diseño permite mantener separación entre desarrollo, entrenamiento e inferencia, y proporciona una base reproducible para evolucionar hacia un esquema productivo con CI/CD, monitoreo, autenticación, HTTPS, gestión de secretos y versionamiento formal de modelos.
