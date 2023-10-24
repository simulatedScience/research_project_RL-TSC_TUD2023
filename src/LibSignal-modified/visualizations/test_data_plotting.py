"""
This module provides tools to summarize and plot test

Authors: Sebastian Jost & GPT-4 (24.10.2023)
"""

import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np

from data_reader import read_and_group_test_data

def plot_averaged_data_with_range(segregated_data, x_param, y_param):
    """
    Plot the averaged metric for each group along with its range.
    
    Args:
    - segregated_data (dict): Dictionary grouping data by noise settings.
    - x_param (str): The noise setting for the x-axis ("fc", "nc", or "nr").
    - y_param (str): The performance metric for the y-axis (e.g., "throughput", "delay").
    """
    average_data = compute_averages(segregated_data)
    segregated_data = segregate_data_by_params(average_data, x_param)
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Define a colormap to differentiate lines
    colormap = plt.get_cmap('tab10')
    colors = np.linspace(0, 1, len(segregated_data))
    
    for idx, (key, group) in enumerate(segregated_data.items()):
        x_values = [run[x_param] for run in group]
        avg_values = [run[y_param]['average'] for run in group]
        min_values = [run[y_param]['min'] for run in group]
        max_values = [run[y_param]['max'] for run in group]

        # Sort data by x values for visualization
        sorted_indices = sorted(range(len(x_values)), key=lambda k: x_values[k])
        x_values = [x_values[i] for i in sorted_indices]
        avg_values = [avg_values[i] for i in sorted_indices]
        min_values = [min_values[i] for i in sorted_indices]
        max_values = [max_values[i] for i in sorted_indices]

        other_params = [param for param in ['fc', 'nc', 'nr'] if param != x_param]
        label = f"{other_params[0]}={key[0]}, {other_params[1]}={key[1]}"
        plt.fill_between(x_values, min_values, max_values, color=colormap(colors[idx]), alpha=0.2)
        plt.plot(x_values, avg_values, 'o-', label=label, color=colormap(colors[idx]))
    
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(f'{y_param} vs {x_param}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def compute_averages(grouped_data: dict) -> list:
    """
    Compute average and range for each metric within each group.
    
    Args:
    - grouped_data (dict): Dictionary grouping data by noise settings.
    
    Returns:
    - list: A list of dictionaries, each containing settings, averaged metrics, and metric ranges.
    """
    averaged_data = []
    
    for key, metrics in grouped_data.items():
        averaged_run = {
            'fc': key[0],
            'nc': key[1],
            'nr': key[2]
        }
        for metric, values in metrics.items():
            avg_value = sum(values) / len(values)
            min_value = min(values)
            max_value = max(values)
            averaged_run[metric] = {
                'average': avg_value,
                'min': min_value,
                'max': max_value
            }
        
        averaged_data.append(averaged_run)
    
    return averaged_data

def segregate_data_by_params(data, x_param):
    """
    Segregate data based on noise settings other than the chosen x_param.
    
    Args:
    - data (list): A list of dictionaries containing settings, averaged metrics, and metric ranges.
    - x_param (str): The noise setting chosen for the x-axis ("fc", "nc", or "nr").
    
    Returns:
    - dict: A dictionary grouping data by the noise settings not on the x-axis.
    """
    # Define the other parameters not on the x-axis
    other_params = [param for param in ['fc', 'nc', 'nr'] if param != x_param]
    
    # Segregate data based on the other parameters
    segregated_data = {}
    for run in data:
        key = (run[other_params[0]], run[other_params[1]])
        if key not in segregated_data:
            segregated_data[key] = []
        segregated_data[key].append(run)
    
    return segregated_data


def main():
    # Prompt user to select a file
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename()
    # load test data from file
    grouped_data = read_and_group_test_data(filepath)
    # plot averaged data with ranges
    plot_averaged_data_with_range(grouped_data, 'fc', 'throughput')
    plot_averaged_data_with_range(grouped_data, 'fc', 'delay')
    plot_averaged_data_with_range(grouped_data, 'nc', 'throughput')
    plot_averaged_data_with_range(grouped_data, 'nc', 'delay')
    
if __name__ == '__main__':
    main()