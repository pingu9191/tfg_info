from mat_utils import *
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import pandas as pd
import json
import math
from time import sleep

def main():
    
    print("Inicio del programa")
    
    data = loadmat("files/formulair04_silverstone 2019 gp 2024-12-26 23-12-10_Stint_1")

    lap = read_data_from_file_mat(data, "Lap")
    throttle = read_data_from_file_mat(data, "ThrottleRaw")
    brake = read_data_from_file_mat(data, "BrakeRaw")
    lapDist = read_data_from_file_mat(data, "LapDist")
    lapCurrentTime = read_data_from_file_mat(data, "LapCurrentLapTime")
    sessionTime = read_data_from_file_mat(data, "SessionTime")
    lapLastLapTime = read_data_from_file_mat(data, "LapLastLapTime")
    
    for i in range(2, len(sessionTime)-2, 2):
            print("Sí", sessionTime[i+1]-sessionTime[i])
            
    empty = []
    for i in range(len(lapLastLapTime)):
        if lapLastLapTime[i] not in empty:
            empty.append(lapLastLapTime[i])
    print(empty)
    return
    
    
    maxlap = np.max(lap)
    
    tho = []
    bra = []
    lap_dist = []
    lap_current_time = []
    for i in range(2, maxlap):
        tho.append(select_lap_from_data(data, throttle, i))
        bra.append(select_lap_from_data(data, brake, i))
        lap_dist.append(select_lap_from_data(data, lapDist, i))
        lap_current_time.append(select_lap_from_data(data, lapCurrentTime, i))
    
    max_lap_dist = 0
    for i in range(len(lap_dist)):
        if max(lap_dist[i]) > max_lap_dist:
            max_lap_dist = max(lap_dist[i])
    
    track_sections = estimate_track_sections(data)
    print(track_sections)
    for section in track_sections:
        section[0] -= 100
        section[1] += 50
    
    tho_resampled = []
    lapDist_resampled = []
    bra_resampled = []
    lapCurrentTime_resampled = []
    jump = 0.5
    for i in range(len(tho)):
        x, y = normalize_data_by_lapDist(tho[i], lap_dist[i], jump)
        tho_resampled.append(x)
        lapDist_resampled.append(y)
        x, y = normalize_data_by_lapDist(bra[i], lap_dist[i], jump)
        bra_resampled.append(x)
        x, y = normalize_data_by_lapDist(lap_current_time[i], lap_dist[i], jump)
        lapCurrentTime_resampled.append(x)
    
    tho = tho_resampled
    lap_dist = lapDist_resampled
    bra = bra_resampled
    lap_current_time = lapCurrentTime_resampled
    
    for i in range(len(tho)):
        tho[i], _ = complete_data_missing_values(tho[i], lap_dist[i], 0, math.ceil(max_lap_dist), jump)
        bra[i], _ = complete_data_missing_values(bra[i], lap_dist[i], 0, math.ceil(max_lap_dist), jump)
        lap_current_time[i], lap_dist[i] = complete_data_missing_values(lap_current_time[i], lap_dist[i], 0, math.ceil(max_lap_dist), jump)
    
    for i in range(len(tho)):
        print("Tho Size: ", len(tho[i]))
        print("Bra Size: ", len(bra[i]))
        print("LapDist Size: ", len(lap_dist[i]))
        
    return
    
    # Extraemos datos de la curva 1 de la pista
    section = track_sections[0]
    
    section_data_th = extract_data_from_section(lap_dist, tho, section)
    section_data_br = extract_data_from_section(lap_dist, bra, section)
    section_data_lpd = extract_data_from_section(lap_dist, lap_dist, section)
    section_data_lct = extract_data_from_section(lap_dist, lap_current_time, section)
    
    for i in range(len(tho)):
        print("Tho Size: ", len(tho[i]))
    
    for i in range(len(bra)):
        print("Bra Size: ", len(bra[i]))
        
    for i in range(len(lap_dist)):
        print("LapDist Size: ", len(lap_dist[i]))
    
    section_times = []
    
    for i in range(len(section_data_lct)):
        section_times.append(section_data_lct[i][-1] - section_data_lct[i][0])
    
    fastetst_lap = np.argmin(section_times)
        
    print("Fastest lap: ", fastetst_lap)
    
    return
    
    while(1):
        plt.plot(section_data_lpd[fastetst_lap], section_data_th[fastetst_lap], label="Throttle", color="red")
        plt.plot(section_data_lpd[fastetst_lap], section_data_br[fastetst_lap], label="Brake", color="red", linestyle='--')
        random  = np.random.randint(0, len(section_data_th))
        while random == fastetst_lap:
            random = np.random.randint(0, len(section_data_th))
        plt.plot(section_data_lpd[random], section_data_th[random], label="Throttle", color="blue")
        plt.plot(section_data_lpd[random], section_data_br[random], label="Brake", color="blue", linestyle='--')
        plt.legend()
        plt.title("Throttle and Brake")
        plt.xlabel(("Time. Lap difference", -section_times[fastetst_lap]-section_times[random]))
        plt.ylabel("Value")
        plt.show()
        plt.pause(0.5)
    
    print("Fin del programa")

if __name__ == "__main__":
    main()
