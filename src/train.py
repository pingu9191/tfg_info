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
BATCH_SIZE = 16
EPOCHS = 500

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
    mask = None
    for section in track.sections:
        X_series_train, X_series_test, X_scalar_train, X_scalar_test, y_train, y_test, minmax_label_u, minmax_scalar_u, mask = read_telemetry_file(f"{output_path}datasets/dataset{k}.npz", 0, 100, 0.15, mask)
        
        minmax_scalar.append(minmax_scalar_u)
        minmax_label.append(minmax_label_u)
        
        batchs_series.append(X_series_train)
        batchs_series_test.append(X_series_test)
        batchs_scalar.append(X_scalar_train)
        batchs_scalar_test.append(X_scalar_test)
        batchs_label.append(y_train)
        batchs_label_test.append(y_test)
        k += 1
    
    print("Datos cargados y modelos preparados.")
    #for i in [2]:
    for i in range(1,len(track.sections)):
        model = MyModel(len(batchs_series[i][0]), len(batchs_series[i][0][0]))
        print(f"Entrenando modelo para la sección {track.sections[i].name}...")
        model.train_model(batchs_series[i], batchs_scalar[i], batchs_label[i], batch=BATCH_SIZE, epochs=EPOCHS)
        model.model.save(f"{output_path}/models/model_section_{track.sections[i].name}.keras")
        print(f"Modelo para la sección {track.sections[i].name} guardado.")
    print("Modelos entrenados.")
    
    # Save models
    """for model, i in zip(models, range(len(models))):
        model.model.save(f"{output_path}/models/model_section_{track.sections[i].name}.keras")
        print(f"Modelo para la sección {track.sections[i].name} guardado.")"""
        
    print("Fin del programa")
    return

if __name__ == "__main__":
    main()
