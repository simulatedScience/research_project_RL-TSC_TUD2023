"""
A program to plot metrics tracked during training of RL TSC agents using the LibSignal library.

Author: Sebastian Jost & GPT-4 (19.10.2023)
"""
import os

import matplotlib.pyplot as plt

from data_reader import read_rl_training_data, experiment_name, exp_config_from_path

def plot_metric_over_episodes(metric, train_data, test_data, exp_path, subtitle=""):
    """
    Plots the given metric over episodes for both training and test data.

    Parameters:
        metric (str): The metric to plot ('real_avg_travel_time', 'q_loss', etc.)
        train_data (dict): Dictionary containing training data
        test_data (dict): Dictionary containing test data
    """
    sim, method, network, exp_name = exp_config_from_path(exp_path, convert_network=True)
    exp_info = experiment_name(sim, method, network, exp_name)
    if subtitle:
        subtitle += "\n"
    exp_subtitle = subtitle + f"Sim: {sim}, Method: {method}, Network: {network}, Exp: {exp_name}"
    # plt.figure(figsize=(12, 6))
    plt.figure(figsize=(10, 4))
    # set subplot configuration
    plt.gcf().subplots_adjust(left=0.075, bottom=0.125, right=0.98, top=0.82, wspace=None, hspace=None)
    plt.plot(train_data["episode"], train_data[metric], label=f"Train {metric}", marker='o')
    plt.plot(test_data["episode"], test_data[metric], label=f"Test {metric}", marker='x')
    plt.xlabel("Episode")
    plt.ylabel(metric)
    plt.title(f"{metric} over Episodes\n{exp_subtitle}")
    plt.legend()
    plt.grid(True)
    # try to create plots folder
    try:
        os.mkdir(os.path.join(exp_path, 'plots'))
    except FileExistsError:
        pass
    plt.savefig(os.path.join(exp_path, 'plots', f'episodes_{metric}_{exp_info}.png'))
    plt.show()

def read_plot_metrics(filepath: str):
    """
    Reads the RL TSC agent output log file and plots the metrics over episodes.

    Args:
        filename (str): The name of the RL TSC agent output file ([datetime]_DTL.log)
    """
    train_data, test_data, noise_settings = read_rl_training_data(filepath)
    noise_settings_str = f"fc={noise_settings['failure chance']}, tpr={noise_settings['true positive rate']}, fpr={noise_settings['false positive rate']}"
    print(f"Plotting metrics for {filepath}...")
    exp_path = filepath.strip(os.path.basename(filepath))[:-len("logger/") ]
    plot_metric_over_episodes("real_avg_travel_time", train_data, test_data, exp_path, subtitle=noise_settings_str)
    plot_metric_over_episodes("q_loss", train_data, test_data, exp_path, subtitle=noise_settings_str)
    plot_metric_over_episodes("rewards", train_data, test_data, exp_path, subtitle=noise_settings_str)
    plot_metric_over_episodes("queue", train_data, test_data, exp_path, subtitle=noise_settings_str)
    plot_metric_over_episodes("delay", train_data, test_data, exp_path, subtitle=noise_settings_str)
    plot_metric_over_episodes("throughput", train_data, test_data, exp_path, subtitle=noise_settings_str)


if __name__ == "__main__":
    # open filedialog to select file
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename

    # open filedialog to select file
    Tk().withdraw()
    filepath = askopenfilename(title="Select RL TSC agent output file", filetypes=[("Log files", "*DTL.log")])
    read_plot_metrics(filepath)