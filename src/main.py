import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_handler import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
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

    for i in range(number_laps_stint_csv(data)):
        lap, t = select_lap_from_data(data, i)
        if t == LapType.INCOMPLETE_LAP:
            continue
        print("Lap ",i,": ", end=" ")
        print_raw_data_in_minutes(get_lap_time(lap))
        print(t, end=" ")
        suma = 0
        for section in track.sections:
            section_data = filter_data_by_section(lap, section)
            suma += get_section_time(section_data)
        print("Sections ", end=" ")
        print_raw_data_in_minutes(suma)
        print("Size ", len(lap[0]))
 
    print("Fin del programa")

    return

if __name__ == "__main__":
    main()
