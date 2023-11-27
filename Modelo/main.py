from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import uvicorn

# Definir la estructura del cuerpo de la solicitud
class PredictionInput(BaseModel):
    COD_DEPTO: int
    COD_MUNI: int
    SEXO: str
    ZONA: str

# Cargar el modelo entrenado
best_clf = joblib.load('rf_model.pkl')

# Iniciar la aplicación FastAPI
app = FastAPI()

# Configurar Jinja2 para plantillas HTML
templates = Jinja2Templates(directory="templates")

# Configurar archivos estáticos (CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ruta de bienvenida con HTML y estilo CSS
@app.get("/")
async def welcome(request: Request):
    return templates.TemplateResponse("welcome.html", {"request": request})

# Definir una ruta para realizar predicciones
@app.post("/predict")
async def predict(data: PredictionInput):
    # Crear un DataFrame a partir de los datos de entrada
    input_data = pd.DataFrame({
        "COD_DEPTO": [data.COD_DEPTO],
        "COD_MUNI": [data.COD_MUNI],
        "SEXO": [data.SEXO],
        "ZONA": [data.ZONA]
    })

    # Aplicar el mismo preprocesamiento que se hizo antes
    test = pd.get_dummies(input_data, columns=['SEXO', 'ZONA'])
    features_to_check = ["SEXO_NO REPORTA", "ZONA_URBANA", "SEXO_MASCULINO"]

    for feature in features_to_check:
        if feature not in test.columns:
            test[feature] = False

    # Seleccionar las características necesarias
    test = test[["COD_DEPTO", "COD_MUNI", "SEXO_MASCULINO", "SEXO_NO REPORTA", "ZONA_URBANA"]]

    # Realizar la predicción usando el modelo cargado
    prediction = best_clf.predict(test)

    # Devolver la predicción como respuesta
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
