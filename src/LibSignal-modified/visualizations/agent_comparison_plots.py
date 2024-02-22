"""
This module provides functions to directly compare different agents on few noise levels using bar charts.
Given a list of different noise settings and agent test logs, this module can extract and group data by agent and noise setting, and then plot the comparison charts for different metrics.

Authors: Sebastian Jost & GPT-4
"""
import os

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
    "exp_2_maxpressure": "maxpressure",
    "exp_8_undisturbed_50": "undisturbed",
    "exp_9_disturbed_100": "disturbed",
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
    cmap = plt.cm.viridis
    colors = [cmap(0.2), cmap(0.8)]
    agents_order = ["maxpressure", "undisturbed", "disturbed"]
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

def main():
    basepath = os.path.join(os.path.dirname("."),
                            "data", "output_data", "tsc")
    # Hardcoded file paths with experiment names and "logger" subfolder
    filepath_maxpresure = r"sumo_maxpressure\sumo1x3\exp_2_maxpressure\logger\2023_10_29-01_05_35_BRF.log" # original test (wrong fpr)
    filepath_undisturbed = r"sumo_presslight\sumo1x3\exp_8_undisturbed_50\logger\2023_10_29-13_41_15_BRF_joined.log" # original test (wrong fpr)
    filepath_disturbed = r"sumo_presslight\sumo1x3\exp_9_disturbed_100\logger\2023_10_30-00_25_30_BRF_joined.log" # original test (wrong fpr)

    # list of file paths for different agents
    filepaths = [filepath_maxpresure, filepath_undisturbed, filepath_disturbed]
    filepaths = [os.path.join(basepath, filepath) for filepath in filepaths]

    # Noise settings to compare
    noise_configs = [
            NoiseSettings(0.0, 1.0, 0.0),
            NoiseSettings(0.1, 0.8, 0.15),
            ]
    
    # Extract and group data, then plot the comparison charts
    throughput_data_grouped = extract_agent_metrics_data(filepaths, noise_configs, "throughput")
    plot_agent_comparison_chart(throughput_data_grouped, noise_configs, "throughput", get_fixedtime_data()["throughput"])
    
    travel_time_data_grouped = extract_agent_metrics_data(filepaths, noise_configs, "travel_time")
    plot_agent_comparison_chart(travel_time_data_grouped, noise_configs, "travel_time", get_fixedtime_data()["travel_time"])

if __name__ == "__main__":
    main()
