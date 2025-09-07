"""
This module provides functions to load data from log files and save it to a CSV file.
"""
import csv
import os
import tkinter as tk
from tkinter import filedialog

def load_files_to_csv(file_paths):
    """ Load data from specified log files and return it as a list of lists (CSV format in memory). """
    data = []
    header = ['exp_identifier', 'agent', 'run_id', 'run_seed', 'fc', 'tpr', 'fpr', 'travel time', 'throughput', 'queue', 'delay', 'rewards']
    data.append(header)

    for agent, file_path in file_paths.items():
        with open(file_path, 'r') as file:
            while True:
                lines = [next(file, None) for _ in range(4)]
                if lines[0] is None:
                    break
                
                if ':' in lines[0] and ':' in lines[1] and ',' in lines[2]:
                    exp_identifier = lines[0].split(':')[1].strip()
                    id_seed_fc_tpr_fpr = lines[1].strip().split(':')[1].split('_')
                    run_id = id_seed_fc_tpr_fpr[0].split('=')[1]
                    run_seed = id_seed_fc_tpr_fpr[1].split('=')[1]
                    fc = id_seed_fc_tpr_fpr[2].split('=')[1]
                    tpr = id_seed_fc_tpr_fpr[3].split('=')[1]
                    fpr = id_seed_fc_tpr_fpr[4].split('=')[1]

                    travel_time_components = lines[2].split(',')
                    travel_time = travel_time_components[0].split('is')[1].strip()
                    rewards = travel_time_components[1].split(':')[1].strip()
                    queue = travel_time_components[2].split(':')[1].strip()
                    delay = travel_time_components[3].split(':')[1].strip()
                    throughput = travel_time_components[4].split(':')[1].strip()

                    data.append([exp_identifier, agent, run_id, run_seed, fc, tpr, fpr, travel_time, throughput, queue, delay, rewards])
    return data

def save_data_to_csv(data, output_file_path):
    """ Save the provided data to a CSV file. """
    with open(output_file_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(data)


def main():
    from agent_comparison_plots import choose_experiments
    from test_data_plotting_combined import get_exp_label
    basepath = os.path.join("data", "output_data", "tsc") 
    list_filepaths = choose_experiments()
    filepaths: dict[str, str] = {get_exp_label(filepath): filepath for filepath in list_filepaths}
    # filepaths["MaxPressure"] = os.path.join(basepath, "sumo_maxpressure/sumo1x3/exp6_1_maxpressure/logger/2024_04_27-12_55_31_BRF.log") # old maxpressure test (original dataset)
    filepaths["MaxPressure"] = os.path.join(basepath, "sumo_maxpressure/sumo1x3/exp_29072025_maxpressure/logger/2025_07_29-18_23_17_BRF.log")
    ### old experiment data (Oct. 2023):
    # # Example of how these functions can be used:
    # filepaths = [
    #     # undisturbed
    #         "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_undisturbed_seed100_eps30_nn32/logger/2024_04_23-15_17_58_BRF.log",
    #         "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_undisturbed_seed200_eps30_nn32/logger/2024_04_23-14_38_23_BRF.log",
    #         "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_undisturbed_seed300_eps30_nn32/logger/2024_04_23-12_31_46_BRF.log",
    #     # disturbed
    #         "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_disturbed_seed100_eps30_nn32/logger/2024_04_23-16_49_18_BRF.log",
    #         "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_disturbed_seed200_eps30_nn32/logger/2024_04_23-16_14_35_BRF.log",
    #         "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_disturbed_seed300_eps30_nn32/logger/2024_04_23-14_01_29_BRF.log",
    #     # maxpressure
    #         "data/output_data/tsc/sumo_maxpressure/sumo1x3/exp6_1_maxpressure/logger/2024_04_27-12_55_31_BRF.log",
    # ]
    data = load_files_to_csv(filepaths)
    root = tk.Tk()
    root.withdraw()
    os.makedirs(os.path.join("data", "output_data", "tsc", "stats"), exist_ok=True)
    output_path = filedialog.asksaveasfilename(initialdir=os.path.join("data", "output_data", "tsc", "stats"), defaultextension=".csv")
    save_data_to_csv(data, output_path)



if __name__ == "__main__":
    main()
