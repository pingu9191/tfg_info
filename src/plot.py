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

file_path = "data/01JWDMAM14G8AT77Y5TW1GCPHE.csv"
track_path = "tracks/tsukuba.json"
output_path = "out/"
JUMP = 500

def main():

    print("Inicio del programa")

    data = read_csv(file_path)
    with open(track_path) as json_file:
        track_data = json.load(json_file)
        track = Track(track_data["track"], track_data["length"], track_data["sections"])

    plt.figure(figsize=(10, 5))
    section = 3
    section = track.sections[section]

    print("Stint laps: ", number_laps_stint_csv(data))
    for i in range(number_laps_stint_csv(data)):
        ret = select_lap_from_data(data, i)
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
            
            lap = filter_data_by_section(lap, section)
            lap1 = filter_data_by_section(lap1, section)
            
            print("Size ", len(lap[0]))
            if t == LapType.VALID_LAP or t == LapType.INCIDENT_LAP:
                plt.plot(read_channel_from_data(lap, "LapDistPct"),
                         read_channel_from_data(lap, "SteeringWheelAngle"), label="Lap "+str(i))
                plt.plot(read_channel_from_data(lap1, "LapDistPct"),
                         read_channel_from_data(lap1, "SteeringWheelAngle"), label="Lap "+str(i))
                plt.title("Throttle Section "+section.name)
                plt.xlabel("Throttle")
                plt.ylabel("Distance")
                plt.legend()
                plt.grid()
                plt.savefig(output_path+"matplot/output_plot.png")
                exit(0)

    print("Fin del programa")

    return

if __name__ == "__main__":
    main()
