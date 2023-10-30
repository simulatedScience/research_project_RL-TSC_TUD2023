import os

import matplotlib.pyplot as plt

from data_reader import read_and_group_test_data, get_fixedtime_data, NoiseSettings#, ABBREVIATIONS
from test_data_plotting import compute_averages


# Define a dictionary for abbreviations
ABBREVIATIONS = {
    "failure_chance": "fc",
    "true_positive_rate": "tpr",
    "false_positive_rate": "fpr"
}
# Define the agent identifiers based on experiment names
AGENT_IDENTIFIERS = {
    "exp_2_maxpressure": "maxpressure",
    "exp_8_disturbed_100": "disturbed",
    "exp_8_undisturbed_50": "undisturbed"
}

def get_noise_config_shortform(noise_config: NoiseSettings) -> str:
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
    - filepaths (List[str]): List of filepaths to the log files.
    - configs (List[namedtuple]): List of parameter configurations.
    - metric (str): Metric to extract data for.
    
    Returns:
    - dict: Data grouped by agent and noise setting.
    """
    data = {} # keys = agents, values = dict where {keys = noise settings, values = metric values}
    
    for filepath in filepaths:
        experiment_name = extract_experiment_name_from_file(filepath)
        agent_name = AGENT_IDENTIFIERS.get(experiment_name, experiment_name)
        grouped_data = read_and_group_test_data(filepath)
        averaged_data = compute_averages(grouped_data)
        
        for config_tuple in configs:
            config_label = get_noise_config_shortform(config_tuple)
            if config_tuple in grouped_data:
                # load previous data for agent
                agent_data: dict = data.get(agent_name, {})
                # summarzie data for config
                avg_data: list[dict] = compute_averages({config_tuple: grouped_data[config_tuple]})
                agent_data[config_tuple] = avg_data[config_tuple][metric]['average']
                data[agent_name] = agent_data
    
    # Add fixedtime data
    fixedtime_value = get_fixedtime_data()[metric]
    data["fixedtime"] = {"fc=0.0": fixedtime_value, "fc=0.1": fixedtime_value}
    
    return data



def plot_agent_comparison_chart(data: dict, noise_configs: list[NoiseSettings], metric: str, fixedtime_value: float):
    cmap = plt.cm.viridis
    colors = [cmap(0.2), cmap(0.8)]
    agents_order = ["maxpressure", "disturbed", "undisturbed"]
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    for i, agent in enumerate(agents_order):
        for j, config in enumerate(noise_configs):
            config_label = get_noise_config_shortform(config)
            bar_label = config_label if i == 0 else ""
            position = i * (len(noise_configs) + 1) * bar_width + j * bar_width
            plt.bar(position, data[agent][f"fc={config.failure_chance}"], color=colors[j], label=bar_label, width=bar_width)
    plt.axhline(fixedtime_value, color='grey', linestyle='--', label=f"FixedTime")
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
    filepath1 = r"sumo_maxpressure\sumo1x3\exp_2_maxpressure\logger\2023_10_29-01_05_35_BRF.log"
    filepath2 = r"sumo_presslight\sumo1x3\exp_8_disturbed_100\logger\2023_10_29-18_23_05_BRF_joined.log"
    filepath3 = r"sumo_presslight\sumo1x3\exp_5_disturbed_100\logger\2023_10_27-22_35_39_BRF.log"
    
    # list of file paths
    filepaths = [filepath1, filepath2, filepath3]
    filepaths = [os.path.join(basepath, filepath) for filepath in filepaths]
    
    # Noise settings to compare
    noise_configs = [
            NoiseSettings(0.0, 1.0, 0.0),
            NoiseSettings(0.1, 0.8, 0.15)]
    
    # Extract and group data, then plot the comparison charts
    throughput_data_grouped = extract_agent_metrics_data(filepaths, noise_configs, "throughput")
    plot_agent_comparison_chart(throughput_data_grouped, noise_configs, "throughput", get_fixedtime_data()["throughput"])
    
    travel_time_data_grouped = extract_agent_metrics_data(filepaths, noise_configs, "travel_time")
    plot_agent_comparison_chart(travel_time_data_grouped, noise_configs, "travel_time", get_fixedtime_data()["travel_time"])

if __name__ == "__main__":
    main()
