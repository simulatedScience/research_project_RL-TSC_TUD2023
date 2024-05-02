"""
This module provides tools to summarize and plot test data of a single agent for many noise settings.
In the plots, each noise parameter ist represented by one color component (Hue, Saturation, brightness) and the x-axis is used for the noise parameter that is not represented by a color component. This allows easy visualization of the effect of different noise settings on the performance metrics.

Authors: Sebastian Jost & GPT-4 (24.10.2023)
"""

import tkinter as tk
from tkinter import filedialog
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import numpy as np

from data_reader import read_and_group_test_data, experiment_name, exp_config_from_path, ABBREVIATIONS, get_fixedtime_data


def plot_averaged_data_with_range(
        segregated_data: dict,
        x_param: str,
        y_param: str,
        exp_path: str,
        y_lim: tuple=None,
        min_max: bool=None,
        ):
    """
    Plot the averaged metric for each group along with its range.
    
    Args:
        segregated_data (dict): Dictionary grouping data by noise settings.
        x_param (str): The noise setting for the x-axis ("failure chance", "true positive rate", or "false positive rate").
        y_param (str): The performance metric for the y-axis (e.g., "throughput", "delay").
        min_max (bool): Whether to use the minimum and maximum values for the range (True) or the standard deviation (False) or no range (None).
    """
    average_data = compute_averages(segregated_data)
    segregated_data = segregate_data_by_params(average_data, x_param)
    # Plotting
    # plt.figure(figsize=(10, 7))
    plt.figure(figsize=(8.8, 10))
    # set subplot configuration
    plt.gcf().subplots_adjust(left=0.075, bottom=0.125, right=0.98, top=0.82, wspace=None, hspace=None)
    
    legend_lines = []
    legend_labels = []
    
    # Define the parameter ranges
    noise_ranges = {
        "fc": (0, 0.17),
        "tpr": (0.55, 1.01),
        "fpr": (0, 0.8),
    }
    # failure_chance_range = (0, 0.2)
    # true_positive_rate_range = (0.6, 1)
    # false_positive_rate_range = (0, 0.65)
    hsv_ranges = [
        (0.0, 1.0),
        (0.4, 1.0),
        (0.5, 1.0),
    ]
    x_param_color = 0.8
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
        
        other_params = [ABBREVIATIONS[param] for param in ['failure chance', 'true positive rate', 'false positive rate'] if param != x_param]
        label = f"{other_params[0]}={key[0]}, {other_params[1]}={key[1]}"
        
        noise_settings = {
            "fc": group[0]["failure chance"],
            "tpr": group[0]["true positive rate"],
            "fpr": group[0]["false positive rate"],
        }
        if other_params[0] == "tpr":
            # swap tpr and fpr
            noise_settings["tpr"], noise_settings["fpr"] = noise_settings["fpr"], noise_settings["tpr"]
            noise_ranges["tpr"], noise_ranges["fpr"] = noise_ranges["fpr"], noise_ranges["tpr"]
            # invert tpr within it's range in noise_ranges
            # noise_settings["tpr"] = noise_ranges["tpr"][1] - noise_settings["tpr"] + noise_ranges["tpr"][0]
        # Determine the color based on the parameters
        color = parameters_to_hsv(
            noise_settings[other_params[0]],
            noise_settings[other_params[1]],
            noise_settings[other_params[1]],
            # noise_settings[ABBREVIATIONS[x_param]],
            noise_ranges[other_params[0]],
            noise_ranges[other_params[1]],
            noise_ranges[other_params[1]],
            # noise_ranges[ABBREVIATIONS[x_param]],
            hsv_ranges=hsv_ranges,
        )
        if other_params[0] == "tpr":
            # swap tpr and fpr
            noise_settings["tpr"], noise_settings["fpr"] = noise_settings["fpr"], noise_settings["tpr"]
            noise_ranges["tpr"], noise_ranges["fpr"] = noise_ranges["fpr"], noise_ranges["tpr"]
        # noise_colors = {
        #     "failure chance": color[0],
        #     "true positive rate": color[1],
        #     "false positive rate": color[2],
        # }
        # color = (color[0], color[1], x_param_color)
        # color = (
        #     noise_colors['failure chance'],
        #     noise_colors['true positive rate'],
        #     noise_colors['false positive rate'],
        # )
        color = mplcolors.hsv_to_rgb(color)

        if min_max is not None:
            plt.fill_between(x_values, min_values, max_values, color=color, alpha=0.2)
        line, = plt.plot(x_values, avg_values, 'o-', label=label, color=color)
        legend_lines.append(line)
        legend_labels.append(label)
    
    # add reference line for fixedtime
    fixedtime_label = 'FixedTime 30s'
    fixedtime_data = get_fixedtime_data()
    fixedtime_line = plt.axhline(y=fixedtime_data[y_param], color='black', linestyle='--', alpha=0.5, label=fixedtime_label)
    legend_lines.append(fixedtime_line)
    legend_labels.append(fixedtime_label)

    # replace underscores with spaces
    x_param_text = x_param.replace('_', ' ')
    y_param_text = y_param.replace('_', ' ')

    sim, method, network, exp_name = exp_config_from_path(exp_path, convert_network=True)
    exp_info = experiment_name(sim, method, network, exp_name)
    exp_subtitle = f"Sim: {sim}, Method: {method}, Network: {network}, Exp: {exp_name}"


    # Reordering the legend entries for row-first filling
    num_cols = 4
    if x_param in ("failure chance", "false positive rate"):
        legend_labels.insert(num_cols, legend_labels.pop(-1))
        legend_lines.insert(num_cols, legend_lines.pop(-1))
        reordered_labels = legend_labels
        reordered_lines = legend_lines
    else:
        reordered_labels = [legend_labels[i::num_cols] for i in range(num_cols)] # sort list into table
        reordered_labels = [label for sublist in reordered_labels for label in sublist] # flatten list
        reordered_lines = [legend_lines[legend_labels.index(label)] for label in reordered_labels] # reorder lines to match labels

    plt.xlabel(x_param_text)
    plt.ylabel(y_param_text)
    # set y limits
    if y_lim is not None:
        plt.ylim(*y_lim)
    plt.title(f'{y_param_text} vs {x_param_text}\n{exp_subtitle}')
    plt.legend(reordered_lines, reordered_labels, loc='best', ncol=num_cols)
    plt.grid(color="#dddddd")
    plt.tight_layout()
    # try to create plots folder
    os.makedirs(os.path.join(exp_path, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(exp_path, 'plots', f'{x_param_text}_{y_param_text}_{exp_info}.png'))
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
            std_value = np.std(values) if len(values) > 1 else np.nan
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


# def parameter_to_rgb(value: float, rgb_min: int=0, rgb_max: int=1) -> int:
#     """
#     Map a parameter value in the range [0, 1] to an RGB value based on the specified min and max.
    
#     Args:
#         value (float): Parameter value in the range [0, 1].
#         rgb_min (int): Minimum RGB value (default is 0).
#         rgb_max (int): Maximum RGB value (default is 255).
    
#     Returns:
#         int: RGB value for the given parameter.
#     """
#     return rgb_min + value * (rgb_max - rgb_min)

def parameters_to_hsv(value1, value2, value3, value1_range, value2_range, value3_range, hsv_ranges=((0, 1), (0, 1), (0, 1))):
    """
    Map the three values to an HSV color based on the given ranges.
    
    Args:
        value1 (float): The first value.
        value2 (float): The second value.
        value3 (float): The third value.
        value1_range (tuple): Min and max value for the first parameter.
        value2_range (tuple): Min and max value for the second parameter.
        value3_range (tuple): Min and max value for the third parameter.
        hsv_range (tuple): Min and max HSV values (default is ((0, 360), (0, 1), (0, 1))).
    
    Returns:
        tuple: HSV color mapped to the given ranges.
    """
    def parameter_to_hsv(value, value_min, value_max, hsv_min, hsv_max):
        """Map a single value to an HSV value based on given ranges."""
        return ((value - value_min) / (value_max - value_min)) * (hsv_max - hsv_min) + hsv_min
    
    h = parameter_to_hsv(value1, *value1_range, *hsv_ranges[0])
    s = parameter_to_hsv(value2, *value2_range, *hsv_ranges[1])
    v = parameter_to_hsv(value3, *value3_range, *hsv_ranges[2])

    return (h, s, v)

def plot_data_for_all_agents(files):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)  # Creating 1 row, 3 columns of subplots with shared Y axis
    for idx, (label, filepath) in enumerate(files.items()):
        data = read_and_group_test_data(filepath)
        # Assuming we have a function that handles the plotting for a given dataset
        plot_averaged_data_with_range(data, 'failure chance', 'travel_time', axs[idx])
        axs[idx].set_title(label)
    
    plt.legend()  # Add a common legend
    plt.tight_layout()
    plt.show()

def parameters_to_color(value1, value2, value3, value1_range, value2_range, value3_range, rgb_range=(0, 1)):
    """
    Map the three values to an RGB color based on the given ranges.
    
    Args:
        value1 (float): The first value.
        value2 (float): The second value.
        value3 (float): The third value.
        value1_range (tuple): Min and max value for the first parameter.
        value2_range (tuple): Min and max value for the second parameter.
        value3_range (tuple): Min and max value for the third parameter.
        rgb_range (tuple): Min and max RGB values (default is (0, 255)).
    
    Returns:
        tuple: RGB color mapped to the given ranges.
    """
    def parameter_to_rgb(value, value_min, value_max, rgb_min, rgb_max):
        """Map a single value to an RGB value based on given ranges."""
        # # map value from range [value_min, value_max] to [0, 1]
        # value01 = (value - value_min) / (value_max - value_min)
        # # map value from range [0, 1] to [rgb_min, rgb_max]
        # color = value01 * (rgb_max - rgb_min) + rgb_min
        # return color
        return ((value - value_min) / (value_max - value_min)) * (rgb_max - rgb_min) + rgb_min
    
    r = parameter_to_rgb(value1, *value1_range, *rgb_range)
    g = parameter_to_rgb(value2, *value2_range, *rgb_range)
    b = parameter_to_rgb(value3, *value3_range, *rgb_range)

    return (r, g, b)


def main():
    # Prompt user to select a file
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(initialdir=r".\data\output_data\tsc\sumo_presslight\sumo1x3")
    print(filepath)
    # load test data from file
    grouped_data = read_and_group_test_data(filepath)
    min_max = None
    exp_path = filepath.strip(os.path.basename(filepath))[:-len("logger/") ]
    # plot averaged data with ranges
    ylim_travel_time = (60, 280)
    ylim_throughput = (1800, 2900)
    plot_averaged_data_with_range(grouped_data, 'failure chance', 'throughput', exp_path, min_max=min_max, y_lim=ylim_throughput)
    plot_averaged_data_with_range(grouped_data, 'failure chance', 'travel_time', exp_path, min_max=min_max, y_lim=ylim_travel_time)
    plot_averaged_data_with_range(grouped_data, 'failure chance', 'queue', exp_path, min_max=min_max)
    # plot_averaged_data_with_range(grouped_data, 'failure chance', 'delay', exp_path, min_max=min_max)
    plot_averaged_data_with_range(grouped_data, 'true positive rate', 'throughput', exp_path, min_max=min_max, y_lim=ylim_throughput)
    plot_averaged_data_with_range(grouped_data, 'true positive rate', 'travel_time', exp_path, min_max=min_max, y_lim=ylim_travel_time)
    plot_averaged_data_with_range(grouped_data, 'true positive rate', 'queue', exp_path, min_max=min_max)
    # plot_averaged_data_with_range(grouped_data, 'true positive rate', 'delay', exp_path, min_max=min_max)
    plot_averaged_data_with_range(grouped_data, 'false positive rate', 'throughput', exp_path, min_max=min_max, y_lim=ylim_throughput)
    plot_averaged_data_with_range(grouped_data, 'false positive rate', 'travel_time', exp_path, min_max=min_max, y_lim=ylim_travel_time)
    plot_averaged_data_with_range(grouped_data, 'false positive rate', 'queue', exp_path, min_max=min_max)
    # plot_averaged_data_with_range(grouped_data, 'false positive rate', 'delay', exp_path, min_max=min_max)
    
if __name__ == '__main__':
    main()
