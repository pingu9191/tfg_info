from mat_utils import *
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import pandas as pd
import json
import math
from time import sleep

file = "ruben/live.csv"

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
    lap = select_lap_from_data_csv(data, sessionTime, i)
    while lap is not None:
        print_raw_data_in_minutes(lap[-1] - lap[0])
        print()
        i += 1
        lap = select_lap_from_data_csv(data, sessionTime, i)
    
    pair = []
    st_f = []
    lasti = []
    last = 0
    
    for i in range(len(sessionTime)-1):
        if lapDistPct[i] > lapDistPct[i+1] and lapDistPct[i]-lapDistPct[i+1] > 950000:
            pair.append(sessionTime[i] - sessionTime[last-1])
            st_f.append([lapDistPct[last-1], lapDistPct[i]])
            last = i+1
            
        if lapLastLapTime[i] not in lasti:
            lasti.append(lapLastLapTime[i])
        
    for i in range(len(pair)):
        print(pair[i])
    
    pair = pair[1:]
    lasti = lasti[1:]
    
    error=0
    print("\n\tEstimatedTime\tlapLastLapTime\tError")
    for i in range(len(pair)):
        print(i,".\t", end="")
        print_raw_data_in_minutes(pair[i])
        print("\t", end=" ")
        print_raw_data_in_minutes(lasti[i])
        print("\t"+str((pair[i]-lasti[i])))
        error += pair[i]-lasti[i]
    
    print("\nError average: ", error/len(pair))
    
    
if __name__ == "__main__":
    main()
