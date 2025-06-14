import sys
import os
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppress TensorFlow warnings
from src.data_handler import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import random
from data.downloader import downloader
from model import MyModel
from utils import Track, LapType
from sklearn.model_selection import train_test_split

file_path = "data/index.txt"
track_path = "tracks/tsukuba.json"
output_path = "out/"
JUMP = 500
BATCH_SIZE = 32

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
        model = MyModel(len(X_series_train[0]), len(X_series_train[0][0]), f"{output_path}/models/model_section_{track.sections[k].name}.keras")
        models.append(model)                            
        k += 1
    
    values = []
    # predict models
    for model, i in zip(models, range(len(models))):
        print(f"Evaluando modelo para la sección {track.sections[i].name}...")
        values.append(model.predict(batchs_series_test[i]))
        print("Medias", np.mean(values[-1]), np.std(values[-1]))
        
    plt.figure(figsize=(10, 5))
    numeros = list(range(len(values[0])))
    permutacion = numeros
    for lap in permutacion[:10]:
        for i in range(len(values)):
            print_raw_data_in_minutes(desnormalize_data(batchs_label_test[i][lap], minmax_label[i][0]
                                         , minmax_label[i][1], minmax_label[i][2]))
        print("")
    print("")
    for lap in permutacion[:10]:
        for i in range(len(values)):
            print_raw_data_in_minutes(desnormalize_data(values[i][lap], minmax_label[i][0]
                                         , minmax_label[i][1], minmax_label[i][2]))
        print("")
    
    mean = []
    for lap in range(math.floor(len(values[i])*0.1)):
        ploteo = []
        mean.append([])
        for i in range(len(values)):
            tt = desnormalize_data(values[i][lap], minmax_label[i][0]
                                             , minmax_label[i][1], minmax_label[i][2]) - desnormalize_data(
                                            batchs_label_test[i][lap], minmax_label[i][0]
                                             , minmax_label[i][1], minmax_label[i][2])
            tt = math.fabs(tt)
            ploteo.append(tt)
            mean[lap].append(tt)	
        plt.plot(range(len(values)), ploteo, label=f"Lap {lap+1}")
    
    for i in range(len(track.sections)):
        print("Media error con media: ", np.mean(np.fabs(batchs_label_test[i] - np.mean(batchs_label_test[i]))))
        print("Media error con modelo: ", np.mean(np.fabs(batchs_label_test[i] - values[i])))
        print("Medias", np.mean(values[i]), np.std(values[i]))
        print(np.std(batchs_label[i]))
    
    values = np.array(mean) 
    values = values.T
    mean = np.mean(values, axis=1)
    plt.plot(range(len(track.sections)), mean, label="Mean", color='black', linewidth=2)                                            

    plt.title("Prediccion de tiempo por sector ")
    plt.xlabel("Sector")
    plt.ylabel("Absolute error with real lap time (s)")
    #plt.legend()
    plt.grid()
    plt.savefig(output_path+"matplot/output_plot.png")

    print("Fin del programa")

    return

if __name__ == "__main__":
    main()
