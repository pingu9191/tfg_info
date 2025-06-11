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
from collections import deque
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, RobustScaler

def read_data_from_file_mat(data: np.ndarray, column_name: str) -> np.ndarray:
    """
    Deprecated, mat files are not supported anymore
    """
    # Asumiendo que los datos están en una estructura de diccionario y "Throttle" es una clave
    throttle_data = data.get(column_name)
    return throttle_data[0][0][1][0]

def read_csv(file: str, delimiter=",") -> np.ndarray:
    """
    Open csv and create numpy array

    Args:
        file (str): path to the csv file

    Returns:
        data (np.ndarray): numpy array with the data
    """
    data = pd.read_csv(file, delimiter=delimiter, dtype={"SessionState": str})

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
    if len(data) == len(utils.model_channels):
        return data[utils.model_channels.index(channel)]
    
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
    """if section.end <= 1:
        section.end = section.end*1000000
        section.start = section.start*1000000"""
    
    indices = (lapDist >= section.start) & (lapDist <= section.end)
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

def select_lap_from_data(data: np.ndarray, lap_number: int) -> list[tuple[np.ndarray, LapType]]:
    """
    Filters the data from a lap and returns the data of the lap
    
    Args:
        data (np.ndarray): numpy array with the data
        lap_number (int): lap number to select
    Returns:
        list[tuple[np.ndarray, LapType]]: list with the data of the lap and the lap type
    """
    
    # Seleccionar los datos de la vuelta 'lap_number'
    laps = read_channel_from_data(data, "Lap")
    lap_indices = np.where(laps == lap_number)[0]

    if (len(lap_indices) == 0):
        return None
    
    ret = []

    if len(lap_indices) != lap_indices[-1] - lap_indices[0] + 1:
        lap_indices2 = deque([])
        while len(lap_indices) != lap_indices[-1] - lap_indices[0] + 1:
            lap_indices2.appendleft(lap_indices[-1])
            lap_indices = lap_indices[:-1]
        ret.extend(select_lap_from_data_aux(data, list(lap_indices2)))
    
    ret.extend(select_lap_from_data_aux(data, lap_indices))
    
    return ret
    
def select_lap_from_data_aux(data: np.ndarray, lap_indices: np.ndarray) -> list[tuple[np.ndarray, LapType]]:
    """
    Recursive auxiliary function to select the lap from the data.
    
    Args:
        data (np.ndarray): numpy array with the data
        lap_number (int): lap number to select
    Returns:
        list[tuple[np.ndarray, LapType]]: list with the data of the lap and the lap type
    """
    
    ret = []
    if len(lap_indices) != lap_indices[-1] - lap_indices[0] + 1:
        lap_indices2 = deque([])
        while len(lap_indices) != lap_indices[-1] - lap_indices[0] + 1:
            lap_indices2.appendleft(lap_indices[-1])
            lap_indices = lap_indices[:-1]
        ret.extend(select_lap_from_data_aux(data, list(lap_indices2)))
        
    filtered_data = []
    if lap_indices[0] == 0:
        lap_indices = lap_indices[1:]
    
    while(data[utils.raw_channels.index("LapDistPct")][lap_indices[-1]] > 1):
        lap_indices = lap_indices[:-1]    
    
    for i in range(len(data)):
        filtered_data.append(data[i][lap_indices[0]-1:lap_indices[-1]+1])

    filtered_data[utils.raw_channels.index("LapDistPct")][0] = 0 # Formalismo
    
    # Eliminar lapDist negativos
    if filtered_data[utils.raw_channels.index("LapDistPct")][1] < 0:
        filtered_data[utils.raw_channels.index("LapDistPct")][1] = filtered_data[utils.raw_channels.index("LapDistPct")][2] / 2
        
    """# Borrar
    menor = 0
    for i in range (len(filtered_data[utils.raw_channels.index("LapDistPct")])-1):
        if filtered_data[utils.raw_channels.index("LapDistPct")][i+1] - filtered_data[utils.raw_channels.index("LapDistPct")][i] > menor and filtered_data[utils.raw_channels.index("LapDistPct")][i+1] - filtered_data[utils.raw_channels.index("LapDistPct")][i] != 0:
            menor = filtered_data[utils.raw_channels.index("LapDistPct")][i+1] - filtered_data[utils.raw_channels.index("LapDistPct")][i]
    print("Menor distancia entre puntos: ", menor)"""

    # Convertir la lista a un array de numpy
    filtered_data = np.array(filtered_data)

    lapType = verify_lapType(filtered_data)

    ret.append((filtered_data, lapType))

    return ret
    
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
    if(lapDist[-1] < 0.99500):
        return LapType.INCOMPLETE_LAP

    surface = read_channel_from_data(data, "PlayerTrackSurface")
    if ((1 in surface) or (-1 in surface)): # -1 offworld, 1 pit, 3 track
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
    lap_time = lap_time[-1]*1000000 - lap_time[0]*1000000
    
    return lap_time

def get_section_time(data: np.ndarray) -> int:
    """
    Get the section time from the data
    
    Args:
        data (np.ndarray): numpy array with the data
    Returns:
        int: section time in microseconds
    """
    
    # Obtener el tiempo de vuelta
    lap_time = read_channel_from_data(data, "SessionTime")
    
    # Convertir el tiempo de vuelta a microsegundos
    lap_time = lap_time[-1]*1000000 - lap_time[0]*1000000
    
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
    if laps is None:
        return None
    
    num_laps = np.max(laps) - np.min(laps) + 1
    return int(num_laps)

def normalize_data(data: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """
    Normaliza los datos a un rango de 0 a 1.
    Usando Z-score
    
    Args:
        data (np.ndarray): array de datos a normalizar
        min (float): valor mínimo del rango
        max (float): valor máximo del rango
    
    Returns:
        np.ndarray: array normalizado, de la misma forma que data
    """
    X = data
    X = np.array(X).flatten()  # Asegurar que es un array 1D
    
    # Paso 1: Normalización Min-Max a [0, 1]
    X_min = np.min(X)
    X_max = np.max(X)
    X_norm = (X - X_min) / (X_max - X_min)
    
    return X_min, X_max, 1.0, X_norm
    
    # Paso 2: Encontrar 'k' tal que la media sea 0.5
    def mean_error(k):
        transformed = X_norm ** k
        return np.abs(np.mean(transformed) - 0.5)
    
    # Optimizar 'k' en el rango (0.1, 10) para evitar overflows
    res = minimize_scalar(mean_error, bounds=(0.1, 10), method='bounded')
    k_opt = res.x
    
    # Aplicar la transformación óptima
    X_final = X_norm ** k_opt
    
    return X_min, X_max, k_opt, X_final

def desnormalize_data(data: np.ndarray, min, max, k):
    """
    Desnormaliza los datos a su rango original.
    
    Args:
        data (np.ndarray): array de datos normalizados
        min (float): valor mínimo del rango original
        max (float): valor máximo del rango original
    
    Returns:
        np.ndarray: array desnormalizado, de la misma forma que data
    """
    return data * (max - min) + min
    
    X_denorm_power = data ** (1 / k)
    
    # Paso 2: Invertir Min-Max
    X_denorm = X_denorm_power * (max - min) + min
    
    return X_denorm

def normalize_channel(data: np.ndarray) -> np.ndarray:
    """
    Normaliza todos los canales de data a un rango de -1 a 1.
    
    Args:
        data (np.ndarray): array de forma (n_channels, n_samples)
    
    Returns:
        np.ndarray: array normalizado, de forma (n_channels, n_samples)
    """
    if len(data) != len(utils.model_channels):
        return data
    
    # Normalizar cada canal por su máximo y mínimo
    normalized_data = np.zeros_like(data, dtype=np.float32)
    
    for i, channel_name in enumerate(utils.model_channels):
        channel_data = data[i]
        
        # Evitar división por cero
        min_val = utils.model_channels_limits[channel_name][0]
        max_val = utils.model_channels_limits[channel_name][1]
        if max_val - min_val == 0:
            normalized_data[i] = 0.0
            continue
        
        # Clip
        x_clipped = np.clip(channel_data, min_val, max_val)
        
        # Normalizar al rango
        normalized_data[i] = (x_clipped - min_val) / (max_val - min_val)
        
    
    return normalized_data

def normalize_data_by_lapDistPct(data: np.ndarray, jump: int) -> np.ndarray:
    """
    Normaliza todos los canales de data agrupando por tramos de 'jump' metros en LapDistPct.
    Vectorizado con Pandas para mejor rendimiento.
    
    Args:
        data (np.ndarray): array de forma (n_channels, n_samples)
        jump (int): salto de metros para agrupar (recomendado: 150-500)
    
    Returns:
        np.ndarray: array normalizado, de forma (n_channels, n_bins)
    """
    jump = jump / 1000000
    
    # Índice del canal de LapDistPct
    lap_index = utils.raw_channels.index("LapDistPct")
    lapDist = data[lap_index]

    # Calcular el bin al que pertenece cada muestra
    bins = np.floor(lapDist / jump).astype(int)
    unique_bins = np.unique(bins)
    num_bins = len(unique_bins)

    # Inicializar resultado
    normalized_data = []

    # Normalizamos solo los canales relevantes
    for channel_name in utils.raw_channels:
        if channel_name not in utils.model_channels:
            continue
        channel_index = utils.raw_channels.index(channel_name)
        channel = data[channel_index]

        # Agrupar por bin y hacer la media manualmente
        averages = np.zeros(num_bins)
        for i, b in enumerate(unique_bins):
            mask = bins == b
            if np.any(mask):
                averages[i] = np.mean(channel[mask])
            else:
                averages[i] = 0

        normalized_data.append(averages)

    # Generar LapDistPct centrado en cada bin
    lapDist_center = unique_bins * jump + jump // 2
    lap_index_in_model = utils.model_channels.index("LapDistPct")
    normalized_data[lap_index_in_model] = lapDist_center

    # Convertir a array numpy (cada fila un canal)
    return np.array(normalized_data)

    
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

def borrar_esta_funcion(data: np.ndarray) -> np.ndarray:
    """
    Deprecated function, not used anymore.
    """
    # Borrar esta función
    
    new_data = []
    for channel in utils.model_channels:
        if channel in utils.new_model_channels:
            new_data.append(data[utils.model_channels.index(channel)])
    
    new_data = np.array(new_data)
    
    return new_data