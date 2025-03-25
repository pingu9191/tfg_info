import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import pandas as pd
import json
from time import sleep
import math
from pandas import DataFrame
from datetime import timedelta

def read_data_from_file_mat(data: np.ndarray, column_name: str) -> np.ndarray:
    # Asumiendo que los datos están en una estructura de diccionario y "Throttle" es una clave
    throttle_data = data.get(column_name)
    return throttle_data[0][0][1][0]

def read_channel_from_file_csv(data: DataFrame, channel: str) -> np.ndarray:
    return np.array(getattr(data, channel).values)
    
def filter_data_by_section(info: np.ndarray, lapDist: np.ndarray, section: list) -> np.ndarray:
    # Filtrar los datos de 'throttle' por la sección de la pista
    section_data = info[(lapDist >= section[0]) & (lapDist < section[1])]
    return section_data

def select_lap_from_data_mat(data: np.ndarray, throttle: np.ndarray, lap_number: int) -> np.ndarray:
    # Seleccionar los datos de la vuelta 'lap_number'
    laps = read_data_from_file_mat(data, "Lap")
    
    lap_indices = np.where(laps == lap_number)[0]
    lap_data = throttle[lap_indices[0]:lap_indices[-1]]
    
    return lap_data

def select_lap_from_data_csv(data: np.ndarray, channel: np.ndarray, lap_number: int) -> np.ndarray:
    # Seleccionar los datos de la vuelta 'lap_number'
    laps = read_channel_from_file_csv(data, "Lap")
    lpdist = read_channel_from_file_csv(data, "SessionTime")
    
    lap_indices = np.where(laps == lap_number)[0]
    
    if len(lap_indices) == 0:
        return None
    
    print("LapIndices[-1]", lap_indices[-1])
    
    lap_data = channel[(lap_indices[0]-1):lap_indices[-1]+1]
    print(lpdist[lap_indices[-1]], lpdist[lap_indices[0]-1], lpdist[lap_indices[-1]]- lpdist[lap_indices[0]-1], end=" ")
    print_raw_data_in_minutes(lpdist[lap_indices[-1]]- lpdist[lap_indices[0]-1])
    
    lpdist = lpdist[lap_indices[0]-1:lap_indices[-1]]
    
    return lap_data

def print_raw_data_in_minutes(raw: int):
    
    raw = int(raw)
    td = timedelta(microseconds=raw)
    minutos = td.seconds // 60
    segundos = td.seconds % 60
    miliseconds = td.microseconds // 1000
    
    if (td.microseconds // 100) % 10 >= 5:
        miliseconds += 1
    
    print("%d:%.2d:%.3d" % (minutos, segundos, miliseconds), end=" ")

def number_laps_stint_mat(data: np.ndarray) -> int:
    # Contar el número de vueltas en el 'stint'
    laps = read_data_from_file_mat(data, "Lap")
    num_laps = np.max(laps) - np.min(laps)
    return num_laps

def number_laps_stint_csv(data: np.ndarray) -> int:
    # Contar el número de vueltas en el 'stint'
    laps = read_data_from_file_csv(data, "Lap")
    num_laps = np.max(laps) - np.min(laps)
    return num_laps

def extract_data_from_section(lapDist: np.ndarray, info: list, section: list) -> list:
    # Extraer la información de la sección de la pista
    section_data = []
    for i in range(len(info)):
        section_data.append(filter_data_by_section(np.array(info[i]), np.array(lapDist[i]), section))
        
    return section_data
    
def estimate_track_sections_mat(data: np.ndarray) -> list:
    
    # Thresholds para la detección de secciones de la pista
    threshold_lat = 1.5
    threshold_lat_under = 0.75
    threshold_long = 0.75
    threshold_long_under = 0.45
    minimum_corner_length = 10
    
    # Estimar las secciones de la pista
    lapDist = read_data_from_file_mat(data, "LapDist")
    g_force_lat = read_data_from_file_mat(data, "LatAccel")
    g_force_long = read_data_from_file_mat(data, "LongAccel")
    i = 2
    
    filtered_track_sections = []
    track_sections = []
    
    while(i < number_laps_stint_mat(data)):
        lap = select_lap_from_data_mat(data, lapDist, i)
        g_lat = select_lap_from_data_mat(data, g_force_lat, i)
        g_long = select_lap_from_data_mat(data, g_force_long, i)
        
        # Calcular el índice de la sección de la pista
        # Sección de la pista: [inicio, fin]
        checkpoint = 0
        straight = True
        
        for j in range(len(lap)):            
            if straight:
                if (abs(g_lat[j]) > threshold_lat) or (abs(g_long[j]) > threshold_long):
                    if(int(lap[j]) - checkpoint > 25):
                        track_sections.append([checkpoint, int(lap[j]), "straight"])
                        checkpoint = int(lap[j])
                    else:
                        kk = track_sections.pop(-1)
                        checkpoint = kk[0]
                        
                    straight = False
            else:
                if (abs(g_lat[j]) < threshold_lat_under) and (abs(g_long[j]) < threshold_long_under):
                    if (int(lap[j]) - checkpoint > 10):
                        track_sections.append([checkpoint, int(lap[j]), "corner"])
                        checkpoint = int(lap[j])
                        straight = True
                    else:
                        kk = track_sections.pop(-1)
                        checkpoint = kk[0]
                        
                    straight = True
        
        i += 1
        
    # return track_sections
    
    # Remove all sections with less than threshold length
    for section in track_sections:
        if section[2] == "straight":
            pass
            #if section[1] - section[0] > 25:
                #filtered_track_sections.append(section)
        else:
            if section[1] - section[0] > minimum_corner_length:
                filtered_track_sections.append(section)

    filtered_track_sections.sort(key=lambda x: x[0])
    # return filtered_track_sections
    
    # Create the sections dictionary
    final_sections = []
    while len(filtered_track_sections) > 0:
        sections = []
        sections.append(filtered_track_sections.pop(0))
        while len(filtered_track_sections) > 0:
            new_section = filtered_track_sections.pop(0)
            if abs(new_section[0]) - abs(sections[-1][0]) > 50:
                break
            else:
                sections.append(new_section)
            
        final_sections.append(sections)
        
        
    # return final_sections
    
    # Create promedium of every section
    promedium_sections = []
    
    for sections in final_sections:
        start = 0
        end = 0
        n = 0
        for section in sections:
            start += section[0]
            end += section[1]
            n += 1

        promedium_sections.append([start/n, end/n])
    
    # Eliminate section to far away from the promedium
    for sections, promedium in zip(final_sections, promedium_sections):
        for section in sections:
            if abs(section[0] - promedium[0]) > 50:
                sections.remove(section)
            elif abs(section[1] - promedium[1]) > 50:
                sections.remove(section)
                
    # Return the promedium sections
    promedium_sections = []
    
    for sections in final_sections:
        start = 0
        end = 0
        n = 0
        for section in sections:
            start += section[0]
            end += section[1]
            n += 1

        promedium_sections.append([int(start/n), int(end/n), "corner"])

    return promedium_sections

def normalize_data_by_lapDist(data: np.ndarray, lapDist: np.ndarray, jump: int) -> tuple[np.ndarray, np.ndarray]:
    
    if len(data) != len(lapDist):
        return None
    
    # Remove the last element if the lapDist is decreasing (error in the data)    
    while lapDist[-1] < lapDist[-2]:
        lapDist = lapDist[:-1]
        data = data[:-1]
    
    # Resample data to have the same size
    new_data = []
    new_lapDist = []
    new_distance = math.floor(lapDist[0])
    
    while lapDist[0] - new_distance > jump/2:
        new_distance += jump
        
    value = data[0]
    count = 1
    i=1

    while i < len(data):
        while math.fabs(lapDist[i] - new_distance) <= jump/2 and i < len(data):
            value += data[i]
            count += 1
            i += 1
            if i == len(data):
                break
            
        new_data.append(value/count)
        new_lapDist.append(new_distance)
        
        if i == len(data):
            break
        
        new_distance = new_distance + jump
        value = 0
        count = 0
        
    return new_data, new_lapDist

def complete_data_missing_values(data: np.ndarray, lapDist: np.ndarray, start: int, end: int, jump: int) -> tuple[np.ndarray, np.ndarray]:
    
    if len(data) != len(lapDist):
        return None
    
    value = lapDist[0] - jump
    while value >= start:
        lapDist = [value] + lapDist
        data = [data[0]] + data
        value -= jump
        
    value = lapDist[-1] + jump
    while value <= end:
        lapDist = lapDist + [value]
        data = data + [data[-1]]
        value += jump
            
    return data, lapDist
