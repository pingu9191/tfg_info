import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow warnings
from src.data_handler import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from model import Model
from utils import Track, LapType

file_path = "tfg_info/data/live.csv"
track_path = "tfg_info/tracks/laguna_seca.json"
output_path = "tfg_info/out/"
JUMP = 275

def main():

    print("Inicio del programa")

    data = read_csv(file_path)
    with open(track_path) as json_file:
        track_data = json.load(json_file)
        track = Track(track_data["track"], track_data["length"], track_data["sections"])

    models = []

    for i in range(number_laps_stint_csv(data)):
        lap, t = select_lap_from_data(data, i)
        time = get_lap_time(lap)
        time = np.array((time,))
        print("Lap ",i,": ", end=" ")
        print_raw_data_in_minutes(get_lap_time(lap))
        print(t, end=" ")
        
        if t == LapType.INCOMPLETE_LAP:
            continue
        
        lap = normalize_data_by_lapDistPct(lap, JUMP)
        
        if len(models) == 0:
            for section in track.sections:
                data_section = filter_data_by_section(lap, section)
                model = Model(data_section, time)
                model.train_model(data_section, time)
                models.append(model)
        else:
            i=0
            for i, section in track.sections:
                data_section = filter_data_by_section(lap, section)
                models[i].train_model(data_section, lap)
                i += 1
            
 
    print("Fin del programa")

    return

if __name__ == "__main__":
    main()
