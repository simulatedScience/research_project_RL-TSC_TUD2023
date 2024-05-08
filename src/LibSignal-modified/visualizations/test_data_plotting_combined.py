"""
This module provides tools to summarize and plot test data of a single agent for many noise settings.
In the plots, each noise parameter ist represented by one color component (Hue, Saturation, brightness) and the x-axis is used for the noise parameter that is not represented by a color component. This allows easy visualization of the effect of different noise settings on the performance metrics.

Authors: Sebastian Jost & GPT-4 (24.10.2023)
"""

# import tkinter as tk
# from tkinter import filedialog
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import numpy as np
import hsluv

from data_reader import read_and_group_test_data, experiment_name, exp_config_from_path, ABBREVIATIONS, get_fixedtime_data


def plot_averaged_data_with_range(
        segregated_data: dict,
        x_param: str,
        y_param: str,
        exp_path: str,
        ax: plt.Axes = None,
        y_lim: tuple = None,
        min_max: bool = None,
        show_legend_and_title: bool = True,
        save_plot: bool = True,
        ) -> tuple[list[str], list[plt.Line2D]]:
    """
    Plot the averaged metric for each group along with its range.
    
    Args:
        segregated_data (dict): Dictionary grouping data by noise settings.
        x_param (str): The noise setting for the x-axis ("failure chance", "true positive rate", or "false positive rate").
        y_param (str): The performance metric for the y-axis (e.g., "throughput", "delay").
        exp_path (str): Path to the experiment folder
        ax (plt.Axes): The axis to plot on (default is None -> create a new figure).
        y_lim (tuple): The y-axis limits (default is None -> fit to data).
        min_max (bool): Whether to use the minimum and maximum values for the range (True) or the standard deviation (False) or no range (None).
        show_legend_and_title (bool): Whether to show the legend and plot title (default is True).
        save_plot (bool): Whether to save the plot as a PNG file (default is True).
    
    Returns:
        (list[str]): List of labels for the legend.
        (list[plt.Line2D]): List of lines for the legend.
    """
    average_data = compute_averages(segregated_data)
    segregated_data = segregate_data_by_params(average_data, x_param)
    # Plotting
    if ax is None:
        # plt.figure(figsize=(10, 7))
        fig = plt.figure(figsize=(8.8, 10))
        fig.tight_layout()
        ax = fig.add_subplot(111)
        # set subplot configuration
        fig.subplots_adjust(left=0.075, bottom=0.125, right=0.98, top=0.82, wspace=None, hspace=None)
        print("Added subplot")
        
    
    legend_lines = []
    legend_labels = []
    
    # Define the parameter ranges
    # noise_ranges = {
    #     "fc": (0, 0.17),
    #     "tpr": (0.55, 1.01),
    #     "fpr": (0, 0.8),
    # }
    noise_ranges = {
        "fc": (0, 0.2),
        "tpr": (0.6, 1.),
        "fpr": (0, 0.65),
    }
    noise_ranges["tpr"] = (1-noise_ranges['tpr'][1], 1-noise_ranges['tpr'][0])
    hsluv_ranges = [
        (0., 240.),
        (0., 100.),
        (35., 85.),
    ]
    # Plot each group of points with the same noise settings except for the x-axis parameter
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
        
        # define list of parameters not on the x-axis (two elements). Use abbreviations
        other_params = [ABBREVIATIONS[param] for param in ['failure chance', 'true positive rate', 'false positive rate'] if param != x_param]
        label = f"{other_params[0]}={key[0]}, {other_params[1]}={key[1]}"
        
        # noise_settings = { # get noise settings for the group
        #     "fc": group[0]["failure chance"],
        #     "tpr": group[0]["true positive rate"],
        #     "fpr": group[0]["false positive rate"],
        # }
        # noise_settings_for_color = [ # get noise settings for the group to calculate line color
        #     group[0]["failure chance"],
        #     1-group[0]["true positive rate"], # invert tpr
        #     group[0]["false positive rate"],
        # ]
        num_cols = 4
        noise_settings_for_color = [idx % num_cols, 1, idx // num_cols]
        noise_ranges_for_color = [(0, 3), (0, 1), (3, 0)]
        # Determine the color based on the parameters

        # values_for_color = [group[0][failure_mode] for failure_mode in ['failure chance', 'true positive rate', 'false positive rate']]
        # noise_ranges_for_color = [noise_ranges["fc"], noise_ranges["tpr"], noise_ranges['fpr']]
        # for i, failure_mode in enumerate(['failure chance', 'true positive rate', 'false positive rate']):
        #     if failure_mode == x_param:
        #         # swap current with middle value
        #         noise_settings_for_color[i], noise_settings_for_color[1] = noise_settings_for_color[1], noise_settings_for_color[i]
        #         # swap corresponding ranges
        #         noise_ranges_for_color[i], noise_ranges_for_color[1] = noise_ranges_for_color[1], noise_ranges_for_color[i]
        #         break
        # noise_settings_for_color[1] = 1
        # noise_ranges_for_color[1] = (0, 1)
            
        color = parameters_to_color(
            noise_settings_for_color,
            noise_ranges_for_color,
            hsluv_ranges
        )
        # if other_params[0] == "tpr":
        #     # swap tpr and fpr
        #     noise_settings["tpr"], noise_settings["fpr"] = noise_settings["fpr"], noise_settings["tpr"]
        #     noise_ranges["tpr"], noise_ranges["fpr"] = noise_ranges["fpr"], noise_ranges["tpr"]
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
        # color = mplcolors.hsv_to_rgb(color)

        if min_max is not None:
            ax.fill_between(x_values, min_values, max_values, color=color, alpha=0.2)
        line, = ax.plot(x_values, avg_values, 'o-', label=label, color=color)
        legend_lines.append(line)
        legend_labels.append(label)
    
    # add reference line for fixedtime
    fixedtime_label = 'FixedTime 30s'
    fixedtime_data = get_fixedtime_data()
    fixedtime_line = ax.axhline(y=fixedtime_data[y_param], color='black', linestyle='--', alpha=0.5, label=fixedtime_label)
    legend_lines.append(fixedtime_line)
    legend_labels.append(fixedtime_label)

    # replace underscores with spaces
    x_param_text = x_param.replace('_', ' ')
    y_param_text = y_param.replace('_', ' ')

    sim, method, network, exp_name = exp_config_from_path(exp_path, convert_network=True)
    exp_info = experiment_name(sim, method, network, exp_name)
    exp_subtitle = f"Method: {method}\nExp: {exp_name}"


    # Reordering the legend entries for row-first filling
    num_cols = 4
    # if x_param in ("false positive rate", ):
    #     legend_labels.insert(num_cols, legend_labels.pop(-1))
    #     legend_lines.insert(num_cols, legend_lines.pop(-1))
    #     reordered_labels = legend_labels
    #     reordered_lines = legend_lines
    # elif x_param in ("failure chance", "true positive rate"): # transpose legend
    reordered_labels = [legend_labels[i::num_cols] for i in range(num_cols)] # sort list into table
    reordered_labels = [label for sublist in reordered_labels for label in sublist] # flatten list
    reordered_lines = [legend_lines[legend_labels.index(label)] for label in reordered_labels] # reorder lines to match labels
    # else:
    #     raise ValueError(f"Eunexpected value encountered for x_param: {x_param}. Expected one of ('failure chance', 'true positive rate', 'false positive rate')")

    ax.set_xlabel(x_param_text)
    ax.set_ylabel(y_param_text)
    # set y limits
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.grid(color="#dddddd")
    if show_legend_and_title:
        ax.set_title(f'{y_param_text} vs {x_param_text}\n{exp_subtitle}')
        ax.legend(reordered_lines, reordered_labels, loc='best', ncol=num_cols)
    # else:
    #     ax.set_title(exp_subtitle)
    if save_plot:
        # try to create plots folder
        os.makedirs(os.path.join(exp_path, 'plots'), exist_ok=True)
        plt.savefig(os.path.join(exp_path, 'plots', f'{ABBREVIATIONS[x_param_text]}_{ABBREVIATIONS[y_param_text]}_{exp_info}.png'))
        # fig.show()
        plt.close()
        plt.clf()
    return reordered_labels, reordered_lines


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


def plot_data_for_all_agents(
        files: dict[str, str],
        x_params: list[str],
        y_params: list[str],
        y_lims: list[tuple[float, float]],
        output_path: str = None,
        ):
    # Creating 1 row, len(files) columns of subplots with shared Y axis
    for y_param, y_lim in zip(y_params, y_lims):
        for x_param in x_params:
            fig, axs = plt.subplots(1, len(files), sharey=True, figsize=(15, 8))
            fig.subplots_adjust(left=0.06, bottom=0.1, right=0.98, top=0.8, wspace=0.05, hspace=0.05)
            for idx, (label, filepath) in enumerate(files.items()):
                exp_path = filepath.strip(os.path.basename(filepath))[:-len("logger/") ]
                data = read_and_group_test_data(filepath)
                # Assuming we have a function that handles the plotting for a given dataset
                legend_labels, legend_lines = plot_averaged_data_with_range(
                    data,
                    x_param,
                    y_param,
                    exp_path=exp_path,
                    ax=axs[idx],
                    y_lim=y_lim,
                    show_legend_and_title=False,
                    save_plot=False,
                    )
                # add label as caption
                axs[idx].text(
                    0.5,
                    -0.085,
                    label,
                    transform=axs[idx].transAxes,
                    fontsize=12,
                    horizontalalignment='center',
                    verticalalignment='top',
                )
        
            x_param_text = x_param.replace('_', ' ')
            y_param_text = y_param.replace('_', ' ')

            sim, method, network, exp_name = exp_config_from_path(exp_path, convert_network=True)
            subtitle = f"Sim: {sim}, Network: {network}"
            fig.suptitle(
                f'{y_param_text} vs {x_param_text}',
                x=0.05,
                y=0.95,
                fontsize=18,
                horizontalalignment='left',
                verticalalignment='top')
            fig.text(0.05, 0.91,
                     subtitle,
                     fontsize=12,
                     horizontalalignment='left',
                     verticalalignment='top')
            
            for idx in range(1, len(axs)):
                axs[idx].set_ylabel('')
            
            # Add a common legend in the top right corner outside the subplots
            fig.legend(legend_lines, legend_labels, ncol=4, loc='upper right')
            # fig.tight_layout()
            if output_path:
                os.makedirs(output_path, exist_ok=True)
                plt.savefig(os.path.join(output_path, f'{ABBREVIATIONS[x_param_text]}_{ABBREVIATIONS[y_param_text]}_combined.png'))
            plt.show()


def parameters_to_rgb_hsluv(param1: float, param2: float, param3: float) -> tuple:
    """
    Given three parameters in range [0,1], return a color using HSLuv color space.
    Each parameter gets mapped to a component of the HSLuv color space, then that color is converted to RGB.
    
    Args:
        param1 (float): First parameter in range [0,1].
        param2 (float): Second parameter in range [0,1].
        param3 (float): Third parameter in range [0,1].

    Returns:
        Tuple[float]: RGB color.
    """
    # Adjust the ranges for better visual differentiation
    h = (param1) * 330 # Hue range [0, 330]
    s = 35 + (param2) * 65 # Saturation range [35, 100]
    l = 20 + (param3) * 60  # Lightness range [20, 80]

    hsluv_color = [h, s, l]
    # rgb_color = hsluv.hsluv_to_rgb(hsluv_color)
    rgb_color = mplcolors.hsv_to_rgb(hsluv_color)
    return tuple(rgb_color)

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

def parameters_to_hsluv(
        values: list[float],
        value_ranges: list[tuple[float, float]],
        hsluv_ranges: list[tuple[float, float]] = ((0, 1), (0, 1), (0, 1))) -> tuple[float, float, float]:
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
    def parameter_to_hsv(value, value_min, value_max, color_min, color_max):
        """Map a single value to an HSluv value based on given ranges."""
        return ((value - value_min) / (value_max - value_min)) * (color_max - color_min) + color_min

    hsluv_components = [
        parameter_to_hsv(value, *value_range, *hsluv_range)
        for value, value_range, hsluv_range in zip(values, value_ranges, hsluv_ranges)
    ]
    rgb_color = hsluv.hsluv_to_rgb(hsluv_components)
    return rgb_color

def parameters_to_color(
        values: list[float, float, float],
        value_ranges: list[tuple[float, float]],
        hsvluv_ranges: list[tuple[float, float]] = ((.1, 1), (0, 1), (.3, .9))
        ) -> tuple[float, float, float]:
    """
    Map the three values to an RGB color based on the given ranges.
    
    Args:
        values (list[float]): The three values to map to an RGB color. (should have length 3.)
        value_range (list[tuple[float, float]]): List of tuples with min and max values for each parameter.
        hsvluv_ranges (list[tuple[float, float]]): List of tuples with min and max values for each parameter in HSLuv space -> e.g. limit minimum and maximumum brightness.
    
    Returns:
        tuple: RGB color mapped to the given ranges.
    """
    # mapped_params = [
    #     (value - value_min) / (value_max - value_min) for value, (value_min, value_max) in zip(values, value_ranges)
    # ]
    color: tuple[float, float, float] = parameters_to_hsluv(values, value_ranges, hsvluv_ranges)
    return color


def main():
    # Prompt user to select a file
    # root = tk.Tk()
    # root.withdraw()
    # filepath = filedialog.askopenfilename(initialdir=r".\data\output_data\tsc\sumo_presslight\sumo1x3")
    # print(filepath)
    
    filepaths = {
    # # undisturbed
    #     "Undisturbed, seed=100": "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_undisturbed_seed100_eps30_nn32/logger/2024_04_23-15_17_58_BRF.log", # ** 2.
        "Undisturbed, seed=200": "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_undisturbed_seed200_eps30_nn32/logger/2024_04_23-14_38_23_BRF.log", # *** 1.
    #     "Undisturbed, seed=300": "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_undisturbed_seed300_eps30_nn32/logger/2024_04_23-12_31_46_BRF.log", # * 3.
    # # disturbed
    #     "Disturbed, seed=100": "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_disturbed_seed100_eps30_nn32/logger/2024_04_23-16_49_18_BRF.log", # * 3.
        "Disturbed, seed=200": "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_disturbed_seed200_eps30_nn32/logger/2024_04_23-16_14_35_BRF.log", # *** 1.
    #     "Disturbed, seed=300": "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_disturbed_seed300_eps30_nn32/logger/2024_04_23-14_01_29_BRF.log", # ** 2.
    # maxpressure
        "MaxPressure": "data/output_data/tsc/sumo_maxpressure/sumo1x3/exp6_1_maxpressure/logger/2024_04_27-12_55_31_BRF.log",
    }
    
    plot_data_for_all_agents(
        files=filepaths,
        x_params=["failure chance", "true positive rate", "false positive rate"],
        y_params=["throughput", "travel_time"],
        y_lims=[(2200, 2900), (60, 200)],
        # y_lims=[(1800, 2900), (60, 280)],
        output_path=os.path.join("data", "output_data", "tsc", "stats")
    )

if __name__ == '__main__':
    main()

