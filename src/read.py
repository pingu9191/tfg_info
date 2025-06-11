import sys
import os
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppress TensorFlow warnings
from src.data_handler import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from data.downloader import downloader
from model import MyModel
from utils import Track, LapType

file_path = "data/index.txt"
data_path = "data/models/"
track_path = "tracks/tsukuba.json"
output_path = "out/"
JUMP = 500
BATCH_SIZE = 15

def main():

    print("Inicio del programa")
    with open(track_path) as json_file:
        track_data = json.load(json_file)
        track = Track(track_data["track"], track_data["length"], track_data["sections"])

    files = []
    lenght_batch = []
    
    min_sections = np.array([np.inf] * len(track.sections))
    max_sections = np.zeros(len(track.sections))
    
    file_line = 0
    file = downloader(file_line)
    
    while file is not False:
        
        file_path = "data/" + file
    
        data = read_csv(file_path)
        if number_laps_stint_csv(data) == None:
            print("No laps found in file: ", file_path)
            file_line += 1
            # Remove file
            os.remove(file_path)
            file = downloader(file_line)
            continue

        for i in range(number_laps_stint_csv(data)):
            ret = select_lap_from_data(data, i)
            if ret is None:
                print("No laps found in file: ", file_path, "for lap ", i)
                continue

            for lap, t in ret:

                if (t not in [LapType.VALID_LAP, LapType.INCIDENT_LAP]):
                    print("Skipping lap due to type: ", t)
                    continue
                
                label = get_lap_time(lap)
                lap1 = normalize_data_by_lapDistPct(lap, JUMP)
                
                # Create files and indexes
                if files == []:
                    j = 0
                    for section in track.sections:
                        j += 1
                        data_section = filter_data_by_section(lap1, section)
                        # Create txt file
                       
                        files.append(file)
                        
                        lenght_batch.append(len(data_section[0]))
                
                j = 0
                index = 0
                for section in track.sections:
                    j += 1
                    data_section_nor = filter_data_by_section(lap1, section)
                    data_section = filter_data_by_section(lap, section)
                    if len(data_section_nor) == 0 or len(data_section_nor[0]) == 0:
                        print("No data for section: ", section.name)
                        # Rollback
                        break
                    
                    if len(data_section_nor[0]) != lenght_batch[index]:
                        print(f"Section {section.name} has less data than expected: {len(data_section_nor[0])} not {lenght_batch[index]}")
                        # Rollback
                        break
                    
                    # Write get_section_time to file
                    with open(f"{data_path}/{track.name}/section_{j}_max.txt", mode="a") as file:
                        file.write(f"{get_section_time(data_section)}")
                        for i in range(len(data_section_nor)):
                            file.write(f";{np.max(data_section_nor[i])}")
                        file.write("\n")
                            
                    with open(f"{data_path}/{track.name}/section_{j}_min.txt", mode="a") as file:
                        file.write(f"{get_section_time(data_section)}")
                        for i in range(len(data_section_nor)):
                            file.write(f";{np.min(data_section_nor[i])}")
                        file.write("\n")
                    
                    index += 1
        
        file_line += 1    
        # Remove file
        os.remove(file_path)
        file = downloader(file_line)
    
    print("Max: ", max_sections)
    print("Min: ", min_sections)
    print("Fin del programa")

    return

if __name__ == "__main__":
    main()
