from mat_utils import *
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import pandas as pd
import json

def main():
    
    print("Inicio del programa")
    
    data = loadmat("files/toyotagr86_lagunaseca 2024-11-20 21-35-57_Stint_1")

    lap = read_data_from_file(data, "Lap")
    throttle = read_data_from_file(data, "ThrottleRaw")
    brake = read_data_from_file(data, "BrakeRaw")
    lapDist = read_data_from_file(data, "LapDist")
    maxlap = np.max(lap)
    
    tho = []
    bra = []
    lap_dist = []
    for i in range(2, maxlap):
        tho.append(select_lap_from_data(data, throttle, i))
        bra.append(select_lap_from_data(data, brake, i))
        lap_dist.append(select_lap_from_data(data, lapDist, i))
    
    track_sections = estimate_track_sections(data)
    print(track_sections)
    for section in track_sections:
        section[0] -= 50
        section[1] += 50
        
    plt.figure(figsize=(10, 5))
    
    # Extraemos datos de la curva 1 de la pista
    for section in track_sections:
        section_data_th = extract_data_from_section(lap_dist, tho, section)
        section_data_br = extract_data_from_section(lap_dist, bra, section)
        section_data_lpd = extract_data_from_section(lap_dist, lap_dist, section)
    
        # Lista de colores
        colors = plt.cm.get_cmap('tab10', len(section_data_th))
    
        print(len(section_data_th))
        print(section_data_lpd[0]) 
    
        # Round section_data_lpd to 0 decimal
        for i in range(len(section_data_lpd)):
            section_data_lpd[i] = [round(num, 0) for num in section_data_lpd[i]]
    
        # Create a dictionary with the data
        data_dict = {}

        for lap, dist in zip(section_data_th, section_data_lpd):
            for i in range(len(dist)):
                if dist[i] in data_dict:
                    data_dict[dist[i]].append(lap[i])
                else:
                    data_dict[dist[i]] = [lap[i]]

        print(data_dict)
        data_dict = dict(sorted(data_dict.items()))
        print(data_dict)

        # calculate average of every key
        data_dict_avg = {}
        for key in data_dict:
            data_dict_avg[key] = np.mean(data_dict[key])

        print(data_dict_avg)

        keys = list(data_dict_avg.keys())
        values = list(data_dict_avg.values())

        plt.plot(keys, values, '-', label="Average")
    
    plt.legend()
    plt.title("Throttle and Brake")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()
    
    print("Fin del programa")

if __name__ == "__main__":
    main()
