import sys
import os
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
track_path = "tracks/tsukuba.json"
output_path = "out/"
JUMP = 500
BATCH_SIZE = 15

def main():

    print("Inicio del programa")
    with open(track_path) as json_file:
        track_data = json.load(json_file)
        track = Track(track_data["track"], track_data["length"], track_data["sections"])

    models = []
    lenght_batch = []
    
    batchs_series = []
    batchs_scalar = []
    label_values = []
    
    scalar_values = []
    
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
            j = 0
            for lap, t in ret:
                #print("Lap ",i,".",j,": ", end=" ")
                j += 1
                #print_raw_data_in_minutes(get_lap_time(lap))
                #print(t, end=" ")

                if (t not in [LapType.VALID_LAP, LapType.INCIDENT_LAP]):
                    print("Skipping lap due to type: ", t)
                    continue
                
                label = get_lap_time(lap)
                lap1 = normalize_data_by_lapDistPct(lap, JUMP)
                
                if models == []:
                    for section in track.sections:
                        data_section = filter_data_by_section(lap1, section)
                        # Create model
                        model = MyModel(len(data_section), len(data_section[0]))
                        models.append(model)
                        batchs_series.append([])
                        batchs_scalar.append([])
                        lenght_batch.append(len(data_section[0]))

                time = 0
                index = 0
                lap_sections = []
                scalar_values = []
                for section in track.sections:
                    data_section = filter_data_by_section(lap1, section)
                    data_section_nor = filter_data_by_section(lap, section)
                    if len(data_section) == 0 or len(data_section[0]) == 0:
                        print("No data for section: ", section.name)
                        # Rollback
                        lap_sections = []
                        scalar_values = []
                        time = 0
                        break
                    
                    if len(data_section[0]) != lenght_batch[index]:
                        print(f"Section {section.name} has less data than expected: {len(data_section[0])} not {lenght_batch[index]}")
                        # Rollback
                        lap_sections = []
                        scalar_values = []
                        time = 0
                        break
                    
                    lap_sections.append(data_section)
                    scalar_values.append(time)
                    time += get_section_time(data_section_nor)
                    
                    if track.sections[-1] == section:
                        k = 0
                        label_values.append(label)
                        for lap_section, scalar_value in zip(lap_sections, scalar_values):
                            batchs_series[k].append(lap_section)
                            batchs_scalar[k].append(scalar_value)
                            
                            if len(batchs_series[k]) >= BATCH_SIZE:
                                # Train model
                                models[k].train_model(np.array(batchs_series[k]), np.array(batchs_scalar[k]), np.array(label_values))
                                print("Traning model for section: ", track.sections[k].name)

                                batchs_series[k] = []
                                batchs_scalar[k] = []
                                if k == len(models) - 1:
                                    label_values = []
                            
                            k += 1
                                
                    index += 1
        
        file_line += 1    
        # Remove file
        os.remove(file_path)
        file = downloader(file_line)
    
    # Save models
    i = 0
    for model in models:
        model.model.save(f"{output_path}/models/model_section_{i+1}.h5")
        print(f"Model for section {track.sections[i].name} saved.")
        i += 1
    
    print("Fin del programa")

    return

if __name__ == "__main__":
    main()
