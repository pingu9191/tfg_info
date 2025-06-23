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
from tensorflow.keras.utils import plot_model

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
    score = 0
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
        model = MyModel(len(X_series_train[0]), len(X_series_train[0][0]), 
                        f"{output_path}/models/model_section_{track.sections[k].name}.keras")
        models.append(model)
        k += 1
    
    """plot_model(models[0].model, to_file=output_path+"matplot/model_plot.png", show_shapes=True, show_layer_names=True)
    exit(0)"""
    
    """for i in range(len(track.sections)-1):
        tiempo_tramo = []
        for k in range(len(batchs_scalar[i+1])):
            
            tiempo_tramo.append(desnormalize_data(batchs_scalar[i+1][k], minmax_scalar[i+1][0], minmax_scalar[i+1]     [1],    minmax_scalar[i+1][2]) - desnormalize_data(batchs_scalar[i][k], minmax_scalar[i][0],       minmax_scalar[i][1],    minmax_scalar[i][2]))
        
        tiempo_tramo = np.array(tiempo_tramo)
        print("Sección: ", track.sections[i].name)
        print("Fastest lap: ", end=" ")
        print_raw_data_in_minutes(np.max(tiempo_tramo))
        print("")
        print("Slowest lap: ", end=" ")
        print_raw_data_in_minutes(np.min(tiempo_tramo))
        print("")
        print("Mean: ", end=" ")
        print_raw_data_in_minutes(np.mean(tiempo_tramo))
        print("")
        print("Standard deviation: ", end=" ")
        print_raw_data_in_minutes(np.std(tiempo_tramo))
        print("")
        print("")
    
    exit(0)
    
    # Hacer histograma con kk
    plt.figure(figsize=(10, 6))
    plt.hist(kk, bins=30, color='blue', alpha=0.7)
    plt.title("Distribución de tiempos por vuelta")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Frecuencia")
    plt.grid()
    plt.savefig(output_path+"matplot/histogram.png")
    print("Datos cargados correctamente")
    
    exit(0)"""
    
    values = []
    # predict models
    for model, i in zip(models, range(len(models))):
        print(f"Evaluando modelo para la sección {track.sections[i].name}...")
        values.append(model.predict(batchs_series_test[i], batchs_scalar_test[i]))
        print("Medias", np.mean(values[-1]), np.std(values[-1]))
       
    values = np.array(np.squeeze(values))  # Convertir a array 2D
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
    for lap in range(math.floor(len(values[i])*1)):
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
        #plt.plot(range(len(values)), ploteo, label=f"Lap {lap+1}")
    
    for i in range(len(track.sections)):
        print("Media error con media : ", np.mean(np.fabs(batchs_label_test[i] - np.mean(batchs_label[i]))))
        print("Media error con modelo: ", np.mean(np.fabs(batchs_label_test[i] - values[i])))
        print("Medias", np.mean(values[i]), np.std(values[i]))
        score += ((np.mean(np.fabs(batchs_label_test[i] - np.mean(batchs_label[i])))-np.mean(np.fabs(batchs_label_test[i] - values[i]))) / len(track.sections))
    print("Score:", score) 
    values = np.array(mean) 
    values = values.T
    mean = np.mean(values, axis=1)
    plt.plot(range(len(track.sections)), mean, label="Mean", color='black', linewidth=2)
    
    average = []
    for i in range(len(track.sections)):
        a = desnormalize_data(batchs_label_test[i], minmax_label[i][0]
                                             , minmax_label[i][1], minmax_label[i][2])
        b = desnormalize_data(batchs_label[i], minmax_label[i][0]
                                             , minmax_label[i][1], minmax_label[i][2])
        average.append(np.mean(np.fabs(a - np.mean(b))))
        
    plt.plot(range(len(track.sections)), average, label="Average", color='black', linewidth=2, linestyle='--')                                      

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
