import sys
import os
import copy
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

file_path = "data/index.txt"
data_path = "data/models/"
track_path = "tracks/tsukuba.json"
output_path = "out/"
JUMP = 500
BATCH_SIZE = 15

def main():

    print("Inicio del programa")
    with open(track_path) as json_file:
        track_data = json.load(json_file)
        track = Track(track_data["track"], track_data["length"], track_data["sections"])

    csv_list = ["Label"]
    csv_list.extend(utils.model_channels)
    
    for i in range(len(csv_list)):
        print(f"Channel: ", {csv_list[i]}, end=" ")
        for k in range (track.number_of_sections):
            file_max = f"data/models/{track.name}/section_{k+1}_max.txt"
            file_min = f"data/models/{track.name}/section_{k+1}_min.txt"
            data_max = read_csv(file_max, delimiter=';')
            data_min = read_csv(file_min, delimiter=';')
            print(f"[",{np.min(data_min[i])},",",{np.max(data_max[i])},"]", end=" ")
        print()
            
    print("Fin del programa")

    return

if __name__ == "__main__":
    main()
