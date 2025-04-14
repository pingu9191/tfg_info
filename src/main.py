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

    plt.figure(figsize=(10, 5))
    section = 10
    section = track.sections[section]
    section = TrackSection(section["sector"], 
                           section["start"]/track.length, section["end"]/track.length)

    for i in range(number_laps_stint_csv(data)):
        lap, t = select_lap_from_data(data, i)
        print("Lap ",i,": ", end=" ")
        print_raw_data_in_minutes(get_lap_time(lap))
        print(t, end=" ")
        lap = normalize_data_by_lapDistPct(lap, JUMP)
        print("Size ", len(lap[0]))
        if t == LapType.VALID_LAP or t == LapType.INCIDENT_LAP:
            section_data = filter_data_by_section(lap, section)
            plt.plot(read_channel_from_data(section_data, "ThrottleRaw"), read_channel_from_data(section_data, "LapDistPct"), label="Lap "+str(i))

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
