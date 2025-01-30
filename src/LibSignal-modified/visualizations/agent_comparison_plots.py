"""
This module provides functions to directly compare different agents on few noise levels using bar charts.
Given a list of different noise settings and agent test logs, this module can extract and group data by agent and noise setting, and then plot the comparison charts for different metrics.

Authors: Sebastian Jost & GPT-4
"""
import os
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt

from data_reader import read_and_group_test_data, get_fixedtime_data, NoiseSettings#, ABBREVIATIONS
from test_data_plotting import compute_averages, segregate_data_by_params


# Define a dictionary for abbreviations
ABBREVIATIONS = {
    "failure_chance": "fc",
    "true_positive_rate": "tpr",
    "false_positive_rate": "fpr"
}
# Define the agent identifiers based on experiment names
AGENT_IDENTIFIERS = {
    # "exp3_1_maxpressure": "maxpressure",
    # "exp3_1_undisturbed_100": "undisturbed",
    # "exp3_1_disturbed_100": "disturbed",
    # "exp3_3_disturbed_100": "disturbed",
    "exp6_1_maxpressure": "maxpressure",
    "exp6_disturbed_seed100_eps30_nn32": "disturbed, seed=100",
    "exp6_disturbed_seed200_eps30_nn32": "disturbed, seed=200",
    "exp6_disturbed_seed300_eps30_nn32": "disturbed, seed=300",
    "exp6_undisturbed_seed100_eps30_nn32": "undisturbed, seed=100",
    "exp6_undisturbed_seed200_eps30_nn32": "undisturbed, seed=200",
    "exp6_undisturbed_seed300_eps30_nn32": "undisturbed, seed=300",
}

def get_noise_config_shortform(noise_config: NoiseSettings) -> str:
    """
    Given a noise configuration, returns a short form string representation of the configuration. This is the recommended way to represent noise configurations in legends of plots.

    Args:
        noise_config (NoiseSettings): The noise configuration to represent.

    Returns:
        str: Short form string representation of the noise configuration depending on the ABBREVIATIONS dictionary.
    """
    return ', '.join([f"{ABBREVIATIONS[key]}={getattr(noise_config, key)}" for key in ABBREVIATIONS])



def extract_experiment_name_from_file(filepath: str) -> str:
    """
    Extracts the experiment name from the given filepath.
    
    Args:
    - filepath (str): Path to the log file.
    
    Returns:
    - str: The extracted experiment name.
    """
    filename = os.path.basename(filepath)
    return filename.split("_", 2)[2].rsplit(".", 1)[0]

def extract_agent_metrics_data(filepaths: list[str], 
                               configs: list[NoiseSettings], 
                               metric: str) -> dict:
    """
    Extracts and groups data by agent and noise setting.
    
    Args:
        filepaths (List[str]): List of filepaths to the log files.
        configs (List[namedtuple]): List of parameter configurations.
        metric (str): Metric to extract data for.
    
    Returns:
        dict: Data grouped by agent and noise setting.
    """
    agent_data = {}
    for filepath in filepaths:
        exp_name: str = extract_experiment_name_from_file(filepath)
        exp_name: str = AGENT_IDENTIFIERS.get(exp_name, exp_name)
        grouped_data = read_and_group_test_data(filepath)
        averaged_data_list = compute_averages(grouped_data)
        relevant_data = {}
        # filter segregated data to only include the required configs
        for avg_data in averaged_data_list:
            config_tuple = NoiseSettings(
                avg_data['failure chance'],
                avg_data['true positive rate'],
                avg_data['false positive rate']
            )
            if config_tuple in configs:
                relevant_data[config_tuple] = avg_data
        agent_data[exp_name] = relevant_data
    # Add fixedtime data
    fixedtime_value = get_fixedtime_data()[metric]
    agent_data["fixedtime"] = fixedtime_value

    return agent_data

def extract_experiment_name_from_file(filepath: str) -> str:
    """
    Extracts the experiment name from the given log file.
    
    Args:
    - filepath (str): Filepath to the log file.
    
    Returns:
    - str: Extracted experiment name.
    """
    with open(filepath, 'r') as file:
        # Read the first line of the file to get the experiment name
        line = file.readline()
        experiment_name = line.split(":")[1].split()[0]
    return experiment_name

def plot_agent_comparison_chart(data: dict, noise_configs: list[NoiseSettings], metric: str, fixedtime_value: float):
    n_reps = 8
    cmap = plt.cm.viridis
    colors = [cmap(0.2), cmap(0.8), cmap(0.5)]
    # agents_order = ["maxpressure", "undisturbed", "disturbed"]
    agents_order = list(data.keys())
    agents_order.remove("fixedtime")
    plt.figure(figsize=(8, 6))
    bar_width = 0.35
    for i, agent in enumerate(agents_order):
        for j, config in enumerate(noise_configs):
            config_label = get_noise_config_shortform(config)
            bar_label = config_label if i == 0 else ""
            position = i * (len(noise_configs) + 1) * bar_width + j * bar_width
            plt.bar(position, data[agent][config][metric]["average"], color=colors[j], label=bar_label, width=bar_width)
            # add error bars with standard deviation
            plt.errorbar(position, data[agent][config][metric]["average"], yerr=data[agent][config][metric]["std"], color='black', capsize=3)
    plt.axhline(fixedtime_value, color='grey', linestyle='--', label=f"FixedTime 30s")
    tick_positions = [(i * (len(noise_configs) + 1) + 0.5) * bar_width for i in range(len(agents_order))]
    plt.xticks(tick_positions, agents_order)
    plt.xlabel('Agent')
    plt.ylabel(metric.replace("_", " ").capitalize())
    plt.title(f'{metric.replace("_", " ").capitalize()} Comparison of RL Agents')
    plt.legend()
    plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5)
    plt.grid(axis='x', which='both', linestyle='none')
    plt.tight_layout()
    plt.show()

def select_multiple_directories():
    root = tk.Tk()
    root.withdraw()
    
    parent_dir = filedialog.askdirectory(title="Select Parent Directory")
    if not parent_dir:
        return []
    
    app = DirectorySelectorApp(parent_dir)
    app.run()
    
    return [os.path.join(parent_dir, d) for d in app.selected_dirs]

class DirectorySelectorApp(tk.Tk):
    def __init__(self, parent_dir):
        super().__init__()
        self.title("Select Directories")
        self.configure(bg="#2e2e2e")
        self.parent_dir = parent_dir
        self.selected_dirs = []
        self.create_widgets()
        self.populate_listbox()
        self.adjust_window_size()
    
    def create_widgets(self):
        self.listbox = tk.Listbox(
            self, selectmode=tk.EXTENDED, bg="#3e3e3e", fg="#ffffff",
            selectbackground="#5e5e5e", selectforeground="#ffffff",
            highlightbackground="#2e2e2e", highlightcolor="#5e5e5e",
            relief=tk.FLAT, bd=0
        )
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.select_button = tk.Button(
            self, text="OK", command=self.get_selected_directories,
            bg="#5e5e5e", fg="#ffffff", activebackground="#7e7e7e",
            activeforeground="#ffffff", relief=tk.FLAT, bd=0
        )
        self.select_button.pack(pady=5)
    
    def populate_listbox(self):
        try:
            directories = [d for d in os.listdir(self.parent_dir) if os.path.isdir(os.path.join(self.parent_dir, d))]
            for directory in directories:
                self.listbox.insert(tk.END, directory)
        except Exception:
            pass
    
    def adjust_window_size(self):
        num_items = self.listbox.size()
        height = min(max(200, num_items * 20), 600)
        self.geometry(f"400x{height}")
    
    def get_selected_directories(self):
        self.selected_dirs = [self.listbox.get(i) for i in self.listbox.curselection()]
        self.quit()
        self.destroy()
    
    def run(self):
        self.mainloop()


def get_latest_log_files(exp_directory: str) -> str | None:
    """
    Within an experiment folder, find the latest *BRF.log file, assuming the filename is in the format `yyyy_mm_dd_hh_mm_ss_BRF.log`.

    Args:
    - exp_directory (str): Path to the experiment directory.

    Returns:
    - str | None: Path to the latest log file. None if no log files are found.
    """
    # log files are contained in the `logger` subfolder
    logger_dir = os.path.join(exp_directory, "logger")
    log_files = [f for f in os.listdir(logger_dir) if f.endswith("_BRF.log")]
    if not log_files:
        return None
    # sort log files by date and time
    log_files.sort(reverse=True)
    return os.path.join(logger_dir, log_files[0])

def choose_experiments() -> list[str]:
    """
    let user choose experiment directory paths with file dialog
    in each directory, find the latest *BRF.log file and use that for filepaths.
    
    Returns:
    - list[str]: List of filepaths to the log files.
    """
    root = tk.Tk()
    root.withdraw()
    # let user pick experiment directories
    exp_dirs = select_multiple_directories()
    filepaths = []
    for exp_dir in exp_dirs:
        # get the latest log file in each directory
        latest_log_file = get_latest_log_files(exp_dir)
        if latest_log_file:
            filepaths.append(latest_log_file)
            print(f"Using log file: `{latest_log_file}`")
        else:
            print(f"No log files found in `{exp_dir}`.")

    return filepaths

def main():
    basepath = os.path.join(os.path.dirname("."),
                            "data", "output_data", "tsc")
    # Hardcoded file paths with experiment names and "logger" subfolder
    # filepath_maxpresure = r"sumo_maxpressure\sumo1x3\exp_2_maxpressure\logger\2023_10_29-01_05_35_BRF.log" # original test (wrong fpr)
    # filepath_undisturbed = r"sumo_presslight\sumo1x3\exp_8_undisturbed_50\logger\2023_10_29-13_41_15_BRF_joined.log" # original test (wrong fpr)
    # filepath_disturbed = r"sumo_presslight\sumo1x3\exp_9_disturbed_100\logger\2023_10_30-00_25_30_BRF_joined.log" # original test (wrong fpr)
    # filepath_disturbed = r"sumo_presslight\sumo1x3\exp_9_disturbed_100\logger\2024_02_22-14_06_34_BRF_75reps.log" # default data, 75reps
    # filepath_disturbed = r"sumo_presslight\sumo1x3\exp_9_disturbed_100\logger\2024_02_22-14_45_09_BRF_random_uniform_10reps.log" # random uniform data, 10reps
    # filepath_disturbed = r"sumo_presslight\sumo1x3_short\exp_9_disturbed_100\logger\2024_02_22-15_13_13_BRF.log" # random uniform data, 10reps
    # new experiments 2024_03_25
    # filepath_maxpresure = r"sumo_maxpressure\sumo1x3\exp3_1_maxpressure\logger\2024_03_25-20_19_58_BRF.log"
    # filepath_undisturbed = r"sumo_presslight\sumo1x3\exp3_1_undisturbed_100\logger\2024_03_25-16_18_14_BRF.log"
    # filepath_disturbed = r"sumo_presslight\sumo1x3\exp3_3_disturbed_100\logger\2024_04_11-14_39_25_BRF.log"
    
    filepaths = choose_experiments() + [os.path.join(basepath, "sumo_maxpressure/sumo1x3/exp6_1_maxpressure/logger/2024_04_27-12_55_31_BRF.log")]
    
    # filepaths = [
    # # undisturbed
    #     "sumo_presslight/sumo1x3/exp6_undisturbed_seed100_eps30_nn32/logger/2024_04_23-15_17_58_BRF.log", # ** 2.
    #     "sumo_presslight/sumo1x3/exp6_undisturbed_seed200_eps30_nn32/logger/2024_04_23-14_38_23_BRF.log", # *** 1.
    #     "sumo_presslight/sumo1x3/exp6_undisturbed_seed300_eps30_nn32/logger/2024_04_23-12_31_46_BRF.log", # * 3.
    # # disturbed
    #     "sumo_presslight/sumo1x3/exp6_disturbed_seed100_eps30_nn32/logger/2024_04_23-16_49_18_BRF.log", # *** 1.
    #     "sumo_presslight/sumo1x3/exp6_disturbed_seed200_eps30_nn32/logger/2024_04_23-16_14_35_BRF.log", # ** 2.
    #     "sumo_presslight/sumo1x3/exp6_disturbed_seed300_eps30_nn32/logger/2024_04_23-14_01_29_BRF.log", # * 3.
    # # maxpressure
    #     "sumo_maxpressure/sumo1x3/exp6_1_maxpressure/logger/2024_04_27-12_55_31_BRF.log",
    # ]
    
    # list of file paths for different agents
    # filepaths = [filepath_maxpresure, filepath_undisturbed, filepath_disturbed]
    # filepaths = [os.path.join(basepath, filepath) for filepath in filepaths]

    # Noise settings to compare
    noise_configs = [
            NoiseSettings(0.0, 1.0, 0.0),
            NoiseSettings(0.05, 0.95, 0.15),
            NoiseSettings(0.1, 0.8, 0.15),
            ]
    
    # Extract and group data, then plot the comparison charts
    throughput_data_grouped = extract_agent_metrics_data(filepaths, noise_configs, "throughput")
    plot_agent_comparison_chart(throughput_data_grouped, noise_configs, "throughput", get_fixedtime_data()["throughput"])
    
    travel_time_data_grouped = extract_agent_metrics_data(filepaths, noise_configs, "travel_time")
    plot_agent_comparison_chart(travel_time_data_grouped, noise_configs, "travel_time", get_fixedtime_data()["travel_time"])

if __name__ == "__main__":
    main()
