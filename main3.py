from mat_utils import *
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import pandas as pd
import json
import math
from time import sleep

file = "data/live.csv"
track = "laguna_seca"
JUMP = 300

def main():
    
    print("Inicio del programa")
    
    # open csv file, delimter is ;
    data = pd.read_csv(file, delimiter=";")
    
    # get the columns
    sessionTime = read_channel_from_file_csv(data, "SessionTime")
    lapLastLapTime = read_channel_from_file_csv(data, "LapLastLapTime")
    lapDistPct = read_channel_from_file_csv(data, "LapDistPct")
    
    if len(sessionTime) != len(lapLastLapTime) or len(sessionTime) != len(lapDistPct):
        print("Error: the number of elements in the columns is different")
        return
    
    i = 1
    laps = [0]
    lap = select_lap_from_data_csv(data, sessionTime, i)
    while lap is not None:
        print_raw_data_in_minutes(lap[-1] - lap[0])
        laps.append(lap[-1] - lap[0])
        print()
        i += 1
        lap = select_lap_from_data_csv(data, sessionTime, i)
    
    pair = []
    
    lap_last_lap = read_channel_from_file_csv(data, "LapLastLapTime")
    for i in range(1, len(lap_last_lap)):
        if lap_last_lap[i] not in pair:
            pair.append(lap_last_lap[i])
    
    error=0
    print("\n\tlapLastLapTime\tEstimatedTime\tError")
    for i in range(len(pair)):
        print(i,".\t", end="")
        print_raw_data_in_minutes(pair[i])
        print("\t", end=" ")
        print_raw_data_in_minutes(laps[i])
        print("\t"+str((pair[i]-laps[i])))
        error += pair[i]-laps[i]
    
    print("\nError average: ", error/len(pair))
    
    # open json
    with open("tracks/"+track+".json") as json_file:
        track_json = json.load(json_file)
        
        micro_sector = np.random.choice(track_json["micro_sectors"])
        section = [micro_sector["start"]/track_json["track_length"], 
                           micro_sector["end"]/track_json["track_length"]]
        
        plt.figure(figsize=(10, 5))
        
        for i in range(1, number_laps_stint_csv(data)-1):
            lapDistPct = read_channel_from_file_csv(data, "LapDistPct")
            lapDistPct = select_lap_from_data_csv(data, lapDistPct, i)
            lapDistPct[0] = 0 # Convenio para que el primer punto sea 0
            lap_brake = select_lap_from_data_csv(data, read_channel_from_file_csv(data, "BrakeRaw"), i)
            lap_brake, lapDistPctnormal = normalize_lap_by_lapDist(lap_brake, lapDistPct, JUMP)
            section_brake, section_lapdist = filter_channel_by_section(lap_brake, lapDistPctnormal, section)
            
            # Plot
            plt.plot(section_lapdist, section_brake, label="Brake")
        
        plt.legend()
        plt.show()
        sleep(1)
            
    
if __name__ == "__main__":
    main()
