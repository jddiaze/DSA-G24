import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import plotly.graph_objects as go
import plotly.express as px


import seaborn as sns

import ipywidgets as widgets
from ipywidgets import interact

from IPython.display import display

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


import warnings

# Ignorar todas las advertencias
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, precision_recall_fscore_support



Archivos = "./Criminalidad"
df1 = pd.read_excel(Archivos+"/DELITOS_CONTRA_MEDIO_AMBIENTE.xlsx")

df_homicidios = pd.read_excel(Archivos+"/HOMICIDIO.xlsx")
df_homicidios

df_homicidios['CANTIDAD_CAMBIO'] = np.where(df_homicidios['CANTIDAD'].diff() > 0, 1, 0)

# Dividir los datos en características (X) y la variable objetivo (y)
X = df_homicidios[['COD_DEPTO', 'COD_MUNI', 'SEXO', 'ZONA']]
y = df_homicidios['CANTIDAD_CAMBIO']

# Convertir variables categóricas a variables dummy
X = pd.get_dummies(X, columns=['SEXO', 'ZONA'], drop_first=True)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define el modelo XGBoost
model = XGBClassifier()

# Define el espacio de búsqueda de hiperparámetros
param_grid = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.1, 0.01, 0.05],
    'n_estimators': [100, 200, 300]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_result = grid_search.fit(X_train, y_train)

# Obtén el mejor modelo y sus hiperparámetros
best_model = grid_result.best_estimator_
best_params = grid_result.best_params_

# Realiza predicciones con el mejor modelo en el conjunto de prueba
predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]

# Evalúa el modelo
accuracy = accuracy_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)
logloss = log_loss(y_test, probabilities)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average=None)

# Muestra las métricas de evaluación
print(f"Accuracy: {accuracy}")
print(f"ROC-AUC: {roc_auc}")
print(f"Log Loss: {logloss}")
for cls, prec, rec, f1_score in zip(range(len(precision)), precision, recall, f1):
    print(f"Clase {cls}: Precision={prec}, Recall={rec}, F1-Score={f1_score}")

# Muestra los mejores resultados de la búsqueda en cuadrícula
print("Mejor precisión:", grid_result.best_score_)








#Importe MLFlow para registrar los experimentos, el regresor de bosques aleatorios y la métrica de error cuadrático medio
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# defina el servidor para llevar el registro de modelos y artefactos
#mlflow.set_tracking_uri('http://localhost:5000')
# registre el experimento
experiment = mlflow.set_experiment("XGBoostClassifier")

# Aquí se ejecuta MLflow sin especificar un nombre o id del experimento. MLflow los crea un experimento para este cuaderno por defecto y guarda las características del experimento y las métricas definidas. 
# Para ver el resultado de las corridas haga click en Experimentos en el menú izquierdo. 
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # defina los parámetros del modelo
    n_estimators = 100 
    max_depth = 3
    learning_rate = 0.1
    # Cree el modelo con los parámetros definidos y entrénelo
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_result = grid_search.fit(X_train, y_train)

    # Obtén el mejor modelo y sus hiperparámetros
    best_model = grid_result.best_estimator_
    best_params = grid_result.best_params_

    # Realiza predicciones con el mejor modelo en el conjunto de prueba
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]

    # Evalúa el modelo
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    logloss = log_loss(y_test, probabilities)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average=None)


  
    # Registre los parámetros
    mlflow.log_param("num_trees", n_estimators)
    mlflow.log_param("maxdepth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
  
    # Registre el modelo
    mlflow.sklearn.log_model(best_model, "XGBoostClassifier-model")
  
    # Cree y registre la métrica de interés
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc",roc_auc)
    mlflow.log_metric("logloss",logloss)
    print(accuracy)