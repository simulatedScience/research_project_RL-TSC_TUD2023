"""
A program to plot metrics tracked during training of RL TSC agents using the LibSignal library.

Author: Sebastian Jost & GPT-4 (19.10.2023)
"""
import matplotlib.pyplot as plt

from data_reader import read_rl_training_data

def plot_metric_over_episodes(metric, train_data, test_data):
    """
    Plots the given metric over episodes for both training and test data.

    Parameters:
        metric (str): The metric to plot ('real_avg_travel_time', 'q_loss', etc.)
        train_data (dict): Dictionary containing training data
        test_data (dict): Dictionary containing test data
    """
    plt.figure(figsize=(12, 6))
    plt.plot(train_data["episode"], train_data[metric], label=f"Train {metric}", marker='o')
    plt.plot(test_data["episode"], test_data[metric], label=f"Test {metric}", marker='x')
    plt.xlabel("Episode")
    plt.ylabel(metric)
    plt.title(f"{metric} over Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()

def read_plot_metrics(filename: str):
    """
    Reads the RL TSC agent output log file and plots the metrics over episodes.

    Args:
        filename (str): The name of the RL TSC agent output file ([datetime]_DTL.log)
    """
    train_data, test_data = read_rl_training_data(filename)
    print(f"Plotting metrics for {filename}...")
    plot_metric_over_episodes("real_avg_travel_time", train_data, test_data)
    plot_metric_over_episodes("q_loss", train_data, test_data)
    plot_metric_over_episodes("rewards", train_data, test_data)
    plot_metric_over_episodes("queue", train_data, test_data)
    plot_metric_over_episodes("delay", train_data, test_data)
    plot_metric_over_episodes("throughput", train_data, test_data)


if __name__ == "__main__":
    # open filedialog to select file
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
    
    # open filedialog to select file
    Tk().withdraw()
    filename = askopenfilename(title="Select RL TSC agent output file", filetypes=[("Log files", "*DTL.log")])
    read_plot_metrics(filename)