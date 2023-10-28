"""
This module provides tools to summarize and plot test

Authors: Sebastian Jost & GPT-4 (24.10.2023)
"""

import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np

from data_reader import read_and_group_test_data

def plot_averaged_data_with_range(
        segregated_data: dict,
        x_param: str,
        y_param: str,
        min_max: bool=False):
    """
    Plot the averaged metric for each group along with its range.
    
    Args:
        segregated_data (dict): Dictionary grouping data by noise settings.
        x_param (str): The noise setting for the x-axis ("failure chance", "true positive rate", or "false positive rate").
        y_param (str): The performance metric for the y-axis (e.g., "throughput", "delay").
        min_max (bool): Whether to use the minimum and maximum values for the range (True) or the standard deviation (False).
    """
    average_data = compute_averages(segregated_data)
    segregated_data = segregate_data_by_params(average_data, x_param)
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Define a colormap to differentiate lines
    # colormap = plt.get_cmap('tab10')
    # colors = np.linspace(0, 1, len(segregated_data))
    
    # Plot each group
    for idx, (key, group) in enumerate(segregated_data.items()):
        x_values = [run[x_param] for run in group]
        avg_values = [run[y_param]['average'] for run in group]
        if min_max:
            min_values = [run[y_param]['min'] for run in group]
            max_values = [run[y_param]['max'] for run in group]
        else: # use standard deviation
            min_values = [run[y_param]['average'] - run[y_param]['std'] for run in group]
            max_values = [run[y_param]['average'] + run[y_param]['std'] for run in group]

        # Sort data by x values for visualization
        sorted_indices = sorted(range(len(x_values)), key=lambda k: x_values[k])
        x_values = [x_values[i] for i in sorted_indices]
        avg_values = [avg_values[i] for i in sorted_indices]
        min_values = [min_values[i] for i in sorted_indices]
        max_values = [max_values[i] for i in sorted_indices]

        color = parameters_to_color(
                r=0.4,
                g=key[0],
                b=key[1],
                r_range=(0, 1),
                g_range=(0, 1),
                b_range=(0, 1),
                )
        other_params = [param for param in ['failure chance', 'true positive rate', 'false positive rate'] if param != x_param]
        label = f"{other_params[0]}={key[0]}, {other_params[1]}={key[1]}"
        plt.fill_between(x_values, min_values, max_values, color=color, alpha=0.2)
        plt.plot(x_values, avg_values, 'o-', label=label, color=color)
    
    # replace underscores with spaces
    x_param = x_param.replace('_', ' ')
    y_param = y_param.replace('_', ' ')
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(f'{y_param} vs {x_param}')
    plt.legend(loc='best')
    plt.grid(color="#dddddd")
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
            'failure chance': key[0],
            'true positive rate': key[1],
            'false positive rate': key[2]
        }
        for metric, values in metrics.items():
            avg_value = sum(values) / len(values)
            min_value = min(values)
            max_value = max(values)
            std_value = np.std(values)
            averaged_run[metric] = {
                'average': avg_value,
                'min': min_value,
                'max': max_value,
                'std': std_value,
            }
        
        averaged_data.append(averaged_run)
    
    return averaged_data

def segregate_data_by_params(data, x_param):
    """
    Segregate data based on noise settings other than the chosen x_param.
    
    Args:
    - data (list): A list of dictionaries containing settings, averaged metrics, and metric ranges.
    - x_param (str): The noise setting chosen for the x-axis ("failure chance", "true positive rate", or "false positive rate").
    
    Returns:
    - dict: A dictionary grouping data by the noise settings not on the x-axis.
    """
    # Define the other parameters not on the x-axis
    other_params = [param for param in ['failure chance', 'true positive rate', 'false positive rate'] if param != x_param]
    
    # Segregate data based on the other parameters
    segregated_data = {}
    for run in data:
        key = (run[other_params[0]], run[other_params[1]])
        if key not in segregated_data:
            segregated_data[key] = []
        segregated_data[key].append(run)

    return segregated_data


def parameter_to_rgb(value: float, rgb_min: int=0, rgb_max: int=1) -> int:
    """
    Map a parameter value in the range [0, 1] to an RGB value based on the specified min and max.
    
    Args:
        value (float): Parameter value in the range [0, 1].
        rgb_min (int): Minimum RGB value (default is 0).
        rgb_max (int): Maximum RGB value (default is 255).
    
    Returns:
        int: RGB value for the given parameter.
    """
    return rgb_min + value * (rgb_max - rgb_min)

def parameters_to_color(r, g, b, r_range=(0, 1), g_range=(0, 1), b_range=(0, 1)):
    """
    Map the given parameters to an RGB color based on their respective ranges.
    
    Args:
        r (float): red value in the range [0, 1].
        g (float): green value in the range [0, 1].
        b (float): blue value in the range [0, 1].
        r_range (tuple): Min and max RGB values for fc (default is (0, 255)).
        g_range (tuple): Min and max RGB values for tpr (default is (0, 255)).
        b_range (tuple): Min and max RGB values for fpr (default is (0, 255)).
    
    Returns:
        tuple: RGB color mapped to the given ranges
    """
    r = parameter_to_rgb(r, *r_range)
    g = parameter_to_rgb(g, *g_range)
    b = parameter_to_rgb(b, *b_range)
    
    return (r, g, b)


def main():
    # Prompt user to select a file
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(initialdir=r".\data\output_data\tsc\sumo_presslight\sumo1x3")
    # load test data from file
    grouped_data = read_and_group_test_data(filepath)
    # plot averaged data with ranges
    plot_averaged_data_with_range(grouped_data, 'failure chance', 'throughput')
    plot_averaged_data_with_range(grouped_data, 'failure chance', 'travel_time')
    plot_averaged_data_with_range(grouped_data, 'failure chance', 'queue')
    plot_averaged_data_with_range(grouped_data, 'true positive rate', 'throughput')
    plot_averaged_data_with_range(grouped_data, 'true positive rate', 'travel_time')
    plot_averaged_data_with_range(grouped_data, 'true positive rate', 'queue')
    plot_averaged_data_with_range(grouped_data, 'false positive rate', 'throughput')
    plot_averaged_data_with_range(grouped_data, 'false positive rate', 'travel_time')
    plot_averaged_data_with_range(grouped_data, 'false positive rate', 'queue')
    
if __name__ == '__main__':
    main()