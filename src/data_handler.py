import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from time import sleep
import math
from pandas import DataFrame
from datetime import timedelta
from utils import LapType, TrackSection
import utils

def read_data_from_file_mat(data: np.ndarray, column_name: str) -> np.ndarray:
    """
    Deprecated, mat files are not supported anymore
    """
    # Asumiendo que los datos están en una estructura de diccionario y "Throttle" es una clave
    throttle_data = data.get(column_name)
    return throttle_data[0][0][1][0]

def read_csv(file: str) -> np.ndarray:
    """
    Open csv and create numpy array

    Args:
        file (str): path to the csv file

    Returns:
        data (np.ndarray): numpy array with the data
    """
    data = pd.read_csv(file, delimiter=";")

    # Convert the DataFrame to numpy array
    data = data.to_numpy()
    data = data.T

    return data

def read_channel_from_file_csv(data: DataFrame, channel: str) -> np.ndarray:
    """
    Deprecated function, better use red_channel_from_data() with read_csv()

    Args:
        data (DataFrame): pandas dataframe with the data
        channel (str): channel name

    Returns:
        np.ndarray: numpy array with the data
    """
    return np.array(getattr(data, channel).values)

def read_channel_from_data(data: np.ndarray, channel: str) -> np.ndarray:
    """
    Read a channel from the data
    
    Args:
        data (np.ndarray): numpy array with the data
        channel (str): channel name
    
    Returns:
        np.ndarray: numpy array with the channel
    """
    if len(data) == len(utils.raw_channels):
        return data[utils.raw_channels.index(channel)]
    if len(data) == len(utils.raw_model_channels):
        return data[utils.raw_model_channels.index(channel)]
    
    return None

def filter_data_by_section(data: np.ndarray, section: TrackSection) -> np.ndarray:
    """
    Returns the data of the lap in the section defined by the section

    Args:
        data (np.ndarray): numpy array with the data
        section (list): list with the start and end of the section

    Returns:
        np.ndarray: numpy array with the data in the section        
    """
    new_data = []
    lapdist = read_channel_from_data(data, "LapDistPct")
    for channel in data:
        new_data.append(filter_channel_by_section(channel, lapdist, section))

    # Convertir la lista a un array de numpy
    data = np.array(new_data)
    return data
    
    
def filter_channel_by_section(channel: np.ndarray, lapDist: np.ndarray, section: TrackSection) -> np.ndarray:
    """
    Returns the data of the channel and lapDist in the section defined by the lapDist
    section = [start, end] in meters

    Args:
        channel (np.ndarray): numpy array with the channel
        lapDist (np.ndarray): numpy array with the lap distance of the same instants
        section (list): list with the start and end of the section
    
    Returns:
        np.ndarray: numpy array with the data in the section
    """
    
    # LapDistPct is fixed-point
    if section.end < 1:
        section.end = section.end*1000000
        section.start = section.start*1000000
    
    indices = (lapDist >= section.end) & (lapDist <= section.start)
    if(indices[-1] == False):
        indices = indices | np.roll(indices, shift=1)

    section_data = channel[indices]
    
    return section_data

def select_lap_from_data_mat(data: np.ndarray, throttle: np.ndarray, lap_number: int) -> np.ndarray:
    """
    Deprecated function, mat files are not supported anymore
    
    Args:
        data (np.ndarray): numpy array with the data
        throttle (np.ndarray): numpy array with the throttle data
        lap_number (int): lap number to select
    
    Returns:
        np.ndarray: numpy array with the throttle data of the lap
    """
    # Seleccionar los datos de la vuelta 'lap_number'
    laps = read_data_from_file_mat(data, "Lap")
    
    lap_indices = np.where(laps == lap_number)[0]
    lap_data = throttle[lap_indices[0]:lap_indices[-1]]
    
    return lap_data

def select_lap_from_data(data: np.ndarray, lap_number: int) -> tuple[np.ndarray, LapType]:
    """
    Filters the data from a lap and returns the data of the lap
    
    Args:
        data (np.ndarray): numpy array with the data
        channel (np.ndarray): numpy array with the channel data
        lap_number (int): lap number to select
    Returns:
        np.ndarray: numpy array with the data of the lap
        int: 1 valid lap, 0 incomplete
    """
    
    # Seleccionar los datos de la vuelta 'lap_number'
    laps = read_channel_from_data(data, "Lap")
    lap_indices = np.where(laps == lap_number)[0]

    if (len(lap_indices) == 0):
        return None, LapType.INCOMPLETE_LAP

    # Formalismos de primera vuelta
    while len(lap_indices) != lap_indices[-1] - lap_indices[0] + 1:
        lap_indices = lap_indices[:-1]
    if lap_indices[0] == 0:
        lap_indices = lap_indices[1:]

    filtered_data = []
    for i in range(len(data)):
        filtered_data.append(data[i][lap_indices[0]-1:lap_indices[-1]+1])

    # Convertir la lista a un array de numpy
    filtered_data = np.array(filtered_data)

    lapType = verify_lapType(filtered_data)

    return filtered_data, lapType
    
def verify_lapType(data: np.ndarray) -> LapType:
    """
    Verify the lap type of the data
    
    Args:
        data (np.ndarray): numpy array with the data
    Returns:
        LapType: lap type
    """
    
    # Verificar el tipo de vuelta
    lapType = LapType.VALID_LAP

    # Verificar si la vuelta es incompleta
    lapDist = read_channel_from_data(data, "LapDistPct")
    if(lapDist[-1] < 999500):
        return LapType.INCOMPLETE_LAP

    surface = read_channel_from_data(data, "PlayerTrackSurface")
    if (1 in surface):
        if surface[0] == 3:
            return LapType.INLAP_LAP
        else:
            return LapType.OUTLAP_LAP
    elif(0 in surface):
        return LapType.OFFTRACK_LAP

    incident_count = read_channel_from_data(data, "PlayerCarTeamIncidentCount")
    if incident_count[-1] - incident_count[0] != 0:
        lapType = LapType.INCIDENT_LAP
    
    return lapType

def get_lap_time(data: np.ndarray) -> int:
    """
    Get the lap time from the data
    
    Args:
        data (np.ndarray): numpy array with the data
    Returns:
        int: lap time in microseconds
    """
    
    # Obtener el tiempo de vuelta
    lap_time = read_channel_from_data(data, "SessionTime")
    
    # Convertir el tiempo de vuelta a microsegundos
    lap_time = lap_time[-1] - lap_time[0]
    
    return lap_time

def select_lap_from_channel_csv(data: np.ndarray, channel: np.ndarray, lap_number: int) -> np.ndarray:
    """
    Deprecated function, better use select_lap_from_data() with read_csv()
    
    Args:
        data (np.ndarray): numpy array with the data
        channel (np.ndarray): numpy array with the channel data
        lap_number (int): lap number to select
    Returns:
        np.ndarray: numpy array with the data of the lap
    """
    # Seleccionar los datos de la vuelta 'lap_number'
    laps = read_channel_from_data(data, "Lap")
    lpdist = read_channel_from_data(data, "SessionTime")
    
    lap_indices = np.where(laps == lap_number)[0]
    
    if len(lap_indices) == 0:
        return None
    
    lap_data = channel[(lap_indices[0]-1):lap_indices[-1]+1]
    
    lpdist = lpdist[lap_indices[0]-1:lap_indices[-1]]
    
    return lap_data

def print_raw_data_in_minutes(raw: int):
    """
    Print the raw data lap_time in minutes, seconds and milliseconds
    Args:
        raw (int): raw data lap_time (timedelta in microseconds)
    """
    
    raw = int(raw)
    td = timedelta(microseconds=raw)
    minutos = td.seconds // 60
    segundos = td.seconds % 60
    miliseconds = td.microseconds // 1000
    
    if (td.microseconds // 100) % 10 >= 5:
        miliseconds += 1
    
    print("%d:%.2d:%.3d" % (minutos, segundos, miliseconds), end=" ")

def number_laps_stint_csv(data: np.ndarray) -> int:
    """
    Count the number of laps in the stint
    Args:
        data (np.ndarray): numpy array with the data
    Returns:
        int: number of laps in the stint
    """
    # Contar el número de vueltas en el 'stint'
    laps = read_channel_from_data(data, "Lap")
    num_laps = np.max(laps) - np.min(laps) + 1
    return int(num_laps)

def normalize_data_by_lapDistPct(data: np.ndarray, jump: int) -> np.ndarray:
        """
        Normalize the data by lap distance, using LapDistPct channel.
        Normalize means to average the data in a window of jump meters in order
        to make every lap have the same number of samples.

        Args:
            data (np.ndarray): numpy array with the data
            jump (int): jump in meters (A value between 150 and 500 is recommended)
        Returns:
            np.ndarray: numpy array with the normalized data
        """
        
        lapDist = read_channel_from_data(data, "LapDistPct")
        lapDist[0] = 0 # Formalidad
        new_data = []
        for i in range(len(data)):
            if (utils.raw_channels[i] not in utils.raw_model_channels):
                continue
            ret, x = normalize_channel_lap_by_lapDist(data[i], lapDist, jump)
            new_data.append(ret)

        # Convertir la lista a un array de numpy
        data = np.array(new_data)
        
        return data
    
def normalize_channel_lap_by_lapDist(channel: np.ndarray, lapDist: np.ndarray, jump: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize the channel by lap distance, using LapDistPct channel.
    Normalize means to average the data in a window of jump meters in order
    to make every lap have the same number of samples.
    
    Args:
        channel (np.ndarray): numpy array with the channel
        lapDist (np.ndarray): numpy array with the lap distance of the same instants
        jump (int): jump in meters (A value between 150 and 500 is recommended)
    Returns:
        np.ndarray: numpy array with the normalized channel
        np.ndarray: numpy array with the normalized lap distance
    """
    
    if len(channel) != len(lapDist):
        return None
    
    # Remove the last element if the lapDist is decreasing (error in the data)    
    while lapDist[-1] < lapDist[-2]:
        lapDist = lapDist[:-1]
        channel = channel[:-1]
    
    # Resample data to have the same size
    new_data = []
    new_lapDist = []
    new_distance = math.floor(lapDist[0])
    
    while lapDist[0] - new_distance > jump/2:
        new_distance += jump
        
    value = channel[0]
    count = 1
    i=1

    while i < len(channel):
        while math.fabs(lapDist[i] - new_distance) <= jump/2 and i < len(channel):
            value += channel[i]
            count += 1
            i += 1
            if i == len(channel):
                break
        
        if count == 0:
            new_data.append(value)
        else:
            new_data.append(value/count)
        new_lapDist.append(np.int64(new_distance))
        
        if i == len(channel):
            break
        
        new_distance = new_distance + jump
        value = 0
        count = 0
        
    return np.array(new_data), np.array(new_lapDist)

def complete_data_missing_values(data: np.ndarray, lapDist: np.ndarray, start: int, end: int, jump: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Complete the data with missing values, using LapDistPct channel.
    Complete means to add the data in a window of jump meters in order
    to make every lap have the same number of samples.
    
    Args:
        data (np.ndarray): numpy array with the data
        lapDist (np.ndarray): numpy array with the lap distance of the same instants
        start (int): start of the lap distance in meters
        end (int): end of the lap distance in meters
        jump (int): jump in meters (A value between 150 and 500 is recommended)
    Returns:
        """
    
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