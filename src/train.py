import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppress TensorFlow warnings
from src.data_handler import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from data.downloader import downloader
from model import MyModel
from utils import Track, LapType
from sklearn.model_selection import train_test_split
import utils

file_path = "data/index.txt"
track_path = "tracks/tsukuba.json"
output_path = "out/"
JUMP = 500
BATCH_SIZE = 64
EPOCHS = 30

def main():
    
    np.random.seed(41)  # Semilla fija

    print("Inicio del programa")
    with open(track_path) as json_file:
        track_data = json.load(json_file)
        track = Track(track_data["track"], track_data["length"]
                      , track_data["label_max"], track_data["sections"])

    models = []
    
    batchs_series = []
    batchs_series_test = []
    batchs_scalar = []
    batchs_scalar_test = []
    minmax_scalar = []
    batchs_label = []
    batchs_label_test = []
    minmax_label = []
    
    k = 0
    for section in track.sections:
        data = np.load(f"{output_path}datasets/dataset{k}.npz")
        
        # Normalize X_scalar and y (series is already normalized)
        if k == 0:
            y = data['y']
            Q1 = np.percentile(y, 0, axis=0)
            Q3 = np.percentile(y, 90, axis=0)
            IQR = Q3 - Q1
            mask = ~((y < (Q1 - 1.5 * IQR)) | (y > (Q3 + 1.5 * IQR)))
        
        X_scalar = data['X_scalar']
        X_scalar = X_scalar[mask]  # Apply the same mask to X_scalar
        if X_scalar[0] != 0:
            X_scalar += np.random.uniform(-8333, 8334, size=X_scalar.shape)  # Add noise to X_scalar
        min, max, k_v, X_scalar = normalize_data(X_scalar)
        minmax_scalar.append((min, max, k_v))
        
        y = data['y']
        y = y[mask]  # Apply the same mask to y
        y += np.random.uniform(-8333, 8334, size=y.shape) # Add noise to y
        min, max, k_v, y = normalize_data(y)
        minmax_label.append((min, max, k_v))
        
        X_series = data['X_series']
        X_series = X_series[mask]  # Apply the same mask to X_series
        new_x_series = []
        for i in range(len(X_series)):
            new_x_series.append(borrar_esta_funcion(X_series[i]))
            
        new_x_series = np.array(new_x_series)
        X_series = new_x_series
        
        X_series_train, X_series_test, X_scalar_train, X_scalar_test, y_train, y_test = train_test_split(
                X_series, X_scalar, y, test_size=0.15, random_state=42
        )
        batchs_series.append(X_series_train)
        batchs_series_test.append(X_series_test)
        batchs_scalar.append(X_scalar_train)
        batchs_scalar_test.append(X_scalar_test)
        batchs_label.append(y_train)
        batchs_label_test.append(y_test)
        model = MyModel(len(X_series_train[0]), len(X_series_train[0][0]))
        models.append(model)                            
        k += 1
    
    print("Datos cargados y modelos preparados.")
    for model, i in zip(models, range(len(models))):
        print(f"Entrenando modelo para la sección {track.sections[i].name}...")
        model.train_model(batchs_series[i][:300], batchs_scalar[i][:300], batchs_label[i][:300], batch=BATCH_SIZE, epochs=EPOCHS)
    print("Modelos entrenados.")
    
    # Save models
    for model, i in zip(models, range(len(models))):
        model.model.save(f"{output_path}/models/model_section_{track.sections[i].name}.keras")
        print(f"Modelo para la sección {track.sections[i].name} guardado.")
        
    print("Fin del programa")
    return

if __name__ == "__main__":
    main()
