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

file_path = "data/01JWDM9SPCXE6PJ90A29S09XX7.csv"
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
    section = 0
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
            
            lap = normalize_data_by_lapDistPct(lap, JUMP)
            print("Size ", len(lap[0]))
            if t == LapType.VALID_LAP or t == LapType.INCIDENT_LAP:
                plt.plot(read_channel_from_data(lap, "LapDistPct"),
                         read_channel_from_data(lap, "ThrottleRaw"), label="Lap "+str(i))

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
