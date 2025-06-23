import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_handler import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import math
from time import sleep
from utils import Track, LapType

file_path = "data/01JWRBJ92W37ZT8H13REB87R05 copy.csv"
track_path = "tracks/tsukuba.json"
output_path = "out/"
JUMP = 500

def main():

    print("Inicio del programa")

    data = read_csv(file_path)
    with open(track_path) as json_file:
        track_data = json.load(json_file)
        track = Track(track_data["track"], track_data["length"], 
                      track_data["label_max"], track_data["sections"])

    plt.figure(figsize=(10, 5))
    section = 2
    section = track.sections[section]

    print("Stint laps: ", number_laps_stint_csv(data))
    for i in range(number_laps_stint_csv(data)):
        ret = select_lap_from_data(data, i)
        if ret is None:
            print("No laps found in file: ", file_path, "for lap ", i)
            continue
        j = 0
        for lap, t in ret:
            print("Lap ",i,".",j,": ", end=" ")
            j += 1
            print_raw_data_in_minutes(get_lap_time(lap))
            print(t, end=" ")

            if (t not in [LapType.VALID_LAP, LapType.INCIDENT_LAP]):
                print("Skipping lap due to type: ", t)
                continue
            
            lap1 = normalize_data_by_lapDistPct(lap, JUMP)
            
            time = 0
            
            data_section = filter_data_by_section(lap, section)
            data_section1 = filter_data_by_section(lap1, section)
            if len(data_section) == 0 or len(data_section[0]) == 0:
                print("No data for section: ", section.name)
                continue
            time += get_section_time(data_section)
            
            _, _, _, kk = normalize_data(read_channel_from_data(data_section, "SteeringWheelAngle"))
            plt.plot(read_channel_from_data(data_section, "LapDistPct"),
                     kk, label="Vector Original Normalizado (0, 1)")
            _, _, _, kk = normalize_data(derivate_channel(read_channel_from_data(data_section, "SteeringWheelAngle")))
            plt.plot (read_channel_from_data(data_section, "LapDistPct"),
                         kk, label="Vector Derivado Normalizado (0, 1)")
            
            plt.title("Vector Ángulo de giro del volante en una sección")
            plt.xlabel("Distancia vuelta porcentual")
            plt.ylabel("Valor de Vector")
            plt.legend()
            plt.grid()
            plt.savefig(output_path+"matplot/output_plot.png")

            print("Fin del programa")

            return
                    
            
            print(f"Total time for lap {i}: {time} seconds")
            #lap = normalize_data_by_lapDistPct(lap, JUMP)
            print("Size ", len(lap1[0]))
            

    plt.title("Valor Acelerador "+section.name)
    plt.xlabel("Valor de Acelerador")
    plt.ylabel("Distancia vuelta porcentual")
    plt.legend()
    plt.grid()
    plt.savefig(output_path+"matplot/output_plot.png")

    print("Fin del programa")

    return

if __name__ == "__main__":
    main()
