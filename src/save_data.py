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
BATCH_SIZE = 100000

def main():

    print("Inicio del programa")
    with open(track_path) as json_file:
        track_data = json.load(json_file)
        track = Track(track_data["track"], track_data["length"]
                      , track_data["label_max"], track_data["sections"])

    counter_valid_laps = 0
    counter_not_full_lap = 0
    counter_offtrack_laps = 0
    counter_data_error = 0
    counter_to_slow = 0
    counter_laps = 0

    models = []
    lenght_batch = []
    
    batchs_series = []
    batchs_scalar = []
    batchs_label = []
    
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
                counter_laps += 1
                #print("Lap ",i,".",j,": ", end=" ")
                j += 1
                #print_raw_data_in_minutes(get_lap_time(lap))
                #print(t, end=" ")

                if (t not in [LapType.VALID_LAP, LapType.INCIDENT_LAP]):
                    print("Skipping lap due to type: ", t)
                    if t == LapType.OFFTRACK_LAP:
                        counter_offtrack_laps += 1
                    else:
                        counter_not_full_lap += 1
                    continue
                
                lap1 = normalize_data_by_lapDistPct(lap, JUMP)
                
                if models == []:
                    for section in track.sections:
                        data_section = filter_data_by_section(lap1, section)
                        # Create model
                        #model = MyModel(len(data_section), len(data_section[0]))
                        model = MyModel(len(data_section), len(data_section[0]), f"{output_path}/models/model_section_{section.name}.keras")
                        models.append(model)
                        batchs_series.append([])
                        batchs_scalar.append([])
                        batchs_label.append([])
                        lenght_batch.append(len(data_section[0]))

                time = 0
                index = 0
                lap_sections = []
                scalar_values = []
                label_values = []
                for section in track.sections:
                    data_section = filter_data_by_section(lap1, section)
                    data_section_nor = filter_data_by_section(lap, section)
                    if len(data_section) == 0 or len(data_section[0]) == 0:
                        print("No data for section: ", section.name)
                        # Rollback
                        counter_data_error += 1
                        lap_sections = []
                        scalar_values = []
                        label_values = []
                        time = 0
                        break
                    
                    if len(data_section[0]) != lenght_batch[index]:
                        print(f"Section {section.name} has less data than expected: {len(data_section[0])} not {lenght_batch[index]}")
                        # Rollback
                        counter_data_error += 1
                        lap_sections = []
                        scalar_values = []
                        label_values = []
                        time = 0
                        break
                    
                    if get_lap_time(lap) > track.label_max:
                        print("Lap time is greater than LABEL_MAX, skipping lap.")
                        # Rollback
                        counter_to_slow += 1
                        lap_sections = []
                        scalar_values = []
                        label_values = []
                        time = 0
                        break
                    
                    # Normalizar valores
                    data_section = normalize_data(data_section)
                    lap_sections.append(data_section)
                    
                    scalar_values.append(normalize_time(time, track.label_max))
                    
                    time_nor = get_section_time(lap) - time
                    label_values.append(normalize_time(time_nor, track.label_max))
                    
                    time += get_section_time(data_section_nor)
                    
                    if track.sections[-1] == section:
                        k = 0
                        #label_values.append(label)
                        for lap_section, scalar_value, label_value in zip(lap_sections, scalar_values, label_values):
                            batchs_series[k].append(lap_section)
                            batchs_scalar[k].append(scalar_value)
                            batchs_label[k].append(label_value)
                            
                            k += 1
                        counter_valid_laps += 1
                              
                    index += 1
        
        file_line += 1    
        # Remove file
        os.remove(file_path)
        file = downloader(file_line)
    
    # Save models
    i = 0
    for model in models:
        #model.model.save(f"{output_path}/models/model_section_{i+1}.keras")
        print(f"Model for section {track.sections[i].name} saved.")
        np.savez_compressed(f"out/datasets/dataset{i}.npz", X_series=np.array(batchs_series[i]), X_scalar=np.array(batchs_scalar[i]), y=np.array(batchs_label[i]))
        i += 1
    
    print("Total laps processed: ", counter_laps)
    print(f"Total valid laps:  {counter_valid_laps}, {(counter_valid_laps / counter_laps) * 100:.2f}%")
    print(f"Total laps not full: {counter_not_full_lap}, {(counter_not_full_lap / counter_laps) * 100:.2f}%")
    print(f"Total laps offtrack: {counter_offtrack_laps}, {(counter_offtrack_laps / counter_laps) * 100:.2f}%")
    print(f"Total laps with data error: {counter_data_error}, {(counter_data_error / counter_laps) * 100:.2f}%")
    print(f"Total laps to slow: {counter_to_slow}, {(counter_to_slow / counter_laps) * 100:.2f}%")
    print("Fin del programa")

    return

if __name__ == "__main__":
    main()
