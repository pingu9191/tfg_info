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

file_path = "data/01JWD9Q8GEZCT4Q1EPFZGKR1TK.csv"
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
    section = 0
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
            for section in track.sections:
                data_section = filter_data_by_section(lap, section)
                data_section1 = filter_data_by_section(lap1, section)
                if len(data_section) == 0 or len(data_section[0]) == 0:
                    print("No data for section: ", section.name)
                    continue
                print(f"{section.name}: {get_section_time(data_section)} seconds. Size: {len(data_section1[0])}")
                time += get_section_time(data_section)
                if section.name == "10":
                    plt.plot(read_channel_from_data(data_section, "LapDistPct"),
                             read_channel_from_data(data_section, "ThrottleRaw"), label="Lap "+str(i))
                    
            
            print(f"Total time for lap {i}: {time} seconds")
            #lap = normalize_data_by_lapDistPct(lap, JUMP)
            print("Size ", len(lap1[0]))
            

    plt.title("Throttle Section "+section.name)
    plt.xlabel("Throttle")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid()
    plt.savefig(output_path+"matplot/output_plot.png")

    print("Fin del programa")

    return

if __name__ == "__main__":
    main()
