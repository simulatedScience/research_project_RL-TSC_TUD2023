"""
This module enhances the plotting functionality of test_data_plotting_combined.py
to create a structured grid layout with undisturbed agents on the left, 
disturbed agents on the right, and title, legend and MaxPressure plot in the middle.

Author: Claude & Sebastian Jost (2025-03-16)
"""

import os
from math import floor, ceil
import re

import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np
import hsluv
from pandas.plotting import parallel_coordinates

# Import the necessary functions from the original module
from test_data_plotting_combined import (
    plot_averaged_data_with_range, read_and_group_test_data, 
    experiment_name, exp_config_from_path, ABBREVIATIONS, get_fixedtime_data
)


def create_structured_grid_plot(
        files: dict[str, str],
        x_param: str,
        y_param: str,
        y_lim: tuple[float, float],
        output_path: str = None,
        ):
    """
    Create a structured grid plot with undisturbed agents on the left, 
    disturbed agents on the right, and title, legend, and MaxPressure plot in the middle.
    
    Args:
        files (dict): Dictionary mapping agent labels to filepaths.
        x_param (str): The parameter for the x-axis.
        y_param (str): The parameter for the y-axis.
        y_lim (tuple): Y-axis limits.
        output_path (str): Path to save the output plot.
    """
    # Sort files into undisturbed and disturbed groups
    undisturbed_files = {k: v for k, v in files.items() if "Undisturbed" in k}
    disturbed_files = {k: v for k, v in files.items() if "Disturbed" in k}
    maxpressure_file = {k: v for k, v in files.items() if "MaxPressure" in k}
    
    # Check that we have files for all categories
    if not undisturbed_files:
        print("Warning: No undisturbed files found")
    if not disturbed_files:
        print("Warning: No disturbed files found")
    if not maxpressure_file:
        print("Warning: No MaxPressure file found")
    
    # Parameters for the grid
    n_undisturbed = len(undisturbed_files)
    n_disturbed = len(disturbed_files)
    n_rows = max(ceil(n_undisturbed/2), ceil(n_disturbed/2))
    
    # Create the figure and gridspec
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(n_rows, 5, width_ratios=[1, 1, 1, 1, 1], height_ratios=[1]*n_rows)
    
    # Add a common super title
    x_param_text = x_param.replace('_', ' ')
    y_param_text = y_param.replace('_', ' ')
    
    # Get experiment info from the first file (assuming all files are from the same experiment)
    first_filepath = next(iter(files.values()))
    exp_path = first_filepath.rsplit('/logger/', 1)[0]
    sim, method, network, exp_name = exp_config_from_path(exp_path, convert_network=True)
    
    # Dictionary to hold the axes for each subplot
    all_axes = {}
    first_plot_done = False
    
    # Create outline rectangles for undisturbed and disturbed sections
    # Making them larger to fully encompass the plots
    undisturbed_rect_ax = fig.add_subplot(gs[:, 0:2], frameon=False)
    undisturbed_rect_ax.set_xticks([])
    undisturbed_rect_ax.set_yticks([])
    undisturbed_rect = patches.Rectangle((-0.05, -0.05), 1.1, 1.1, linewidth=1, 
                                         edgecolor='#cccccc', facecolor='none', 
                                         transform=undisturbed_rect_ax.transAxes)
    undisturbed_rect_ax.add_patch(undisturbed_rect)
    undisturbed_rect_ax.text(0.5, 1.02, "Undisturbed Agents", 
                             transform=undisturbed_rect_ax.transAxes,
                             fontsize=14, ha='center', va='bottom')
    
    disturbed_rect_ax = fig.add_subplot(gs[:, 3:5], frameon=False)
    disturbed_rect_ax.set_xticks([])
    disturbed_rect_ax.set_yticks([])
    disturbed_rect = patches.Rectangle((-0.05, -0.05), 1.1, 1.1, linewidth=1, 
                                       edgecolor='#cccccc', facecolor='none', 
                                       transform=disturbed_rect_ax.transAxes)
    disturbed_rect_ax.add_patch(disturbed_rect)
    disturbed_rect_ax.text(0.5, 1.02, "Disturbed Agents", 
                           transform=disturbed_rect_ax.transAxes,
                           fontsize=14, ha='center', va='bottom')
    
    # Create a list to collect all legend lines and labels
    all_legend_lines = []
    all_legend_labels = []

    # Process undisturbed files (left side)
    undisturbed_files_sorted = sorted(undisturbed_files.items(), 
                                     key=lambda x: int(re.search(r'seed=(\d+)', x[0]).group(1)))
    
    for idx, (label, filepath) in enumerate(undisturbed_files_sorted):
        row = idx // 2
        col = idx % 2
        
        # Create axis
        ax = fig.add_subplot(gs[row, col])
        all_axes[(row, col)] = ax
        
        # Read data
        exp_path = filepath.rsplit('/logger/', 1)[0]
        data = read_and_group_test_data(filepath)
        
        # Extract seed number for simplified label
        seed_match = re.search(r'seed=(\d+)', label)
        seed_num = seed_match.group(1) if seed_match else "Unknown"
        simplified_label = f"seed={seed_num}"
        
        # Plot data
        legend_labels, legend_lines = plot_averaged_data_with_range(
            data,
            x_param,
            y_param,
            exp_path=exp_path,
            ax=ax,
            y_lim=y_lim,
            min_max=None,
            show_labels=False,  # Hide all labels, we'll add common ones later
            show_legend_and_title=False,
            save_plot=False,
        )
        
        # Store all legend entries from first plot with data
        if not all_legend_labels and legend_labels:
            all_legend_labels = legend_labels
            all_legend_lines = legend_lines
        
        # Set title as seed number
        ax.set_title(simplified_label, fontsize=10)
        
        # Hide tick labels except for leftmost and bottom plots
        if col != 0:
            ax.set_yticklabels([])
        if row != n_rows-1:
            ax.set_xticklabels([])

    # Process disturbed files (right side)
    disturbed_files_sorted = sorted(disturbed_files.items(), 
                                   key=lambda x: int(re.search(r'seed=(\d+)', x[0]).group(1)))
    
    for idx, (label, filepath) in enumerate(disturbed_files_sorted):
        row = idx // 2
        col = idx % 2 + 3  # Start from column 3
        
        # Create axis
        ax = fig.add_subplot(gs[row, col])
        all_axes[(row, col)] = ax
        
        # Read data
        exp_path = filepath.rsplit('/logger/', 1)[0]
        data = read_and_group_test_data(filepath)
        
        # Extract seed number for simplified label
        seed_match = re.search(r'seed=(\d+)', label)
        seed_num = seed_match.group(1) if seed_match else "Unknown"
        simplified_label = f"seed={seed_num}"
        
        # Plot data
        legend_labels, legend_lines = plot_averaged_data_with_range(
            data,
            x_param,
            y_param,
            exp_path=exp_path,
            ax=ax,
            y_lim=y_lim,
            min_max=None,
            show_labels=False,  # Hide all labels, we'll add common ones later
            show_legend_and_title=False,
            save_plot=False,
        )
        
        # Store all legend entries from first plot with data
        if not all_legend_labels and legend_labels:
            all_legend_labels = legend_labels
            all_legend_lines = legend_lines
            
        # Set title as seed number
        ax.set_title(simplified_label, fontsize=10)
        
        # Hide tick labels except for rightmost and bottom plots
        if col != 4:
            ax.set_yticklabels([])
        if row != n_rows-1:
            ax.set_xticklabels([])

    # Process MaxPressure (middle)
    if maxpressure_file:
        # Create the MaxPressure plot in the middle bottom
        maxp_ax = fig.add_subplot(gs[n_rows-1, 2])
        all_axes[(n_rows-1, 2)] = maxp_ax
        
        maxp_label, maxp_filepath = next(iter(maxpressure_file.items()))
        exp_path = maxp_filepath.rsplit('/logger/', 1)[0]
        data = read_and_group_test_data(maxp_filepath)
        
        plot_averaged_data_with_range(
            data,
            x_param,
            y_param,
            exp_path=exp_path,
            ax=maxp_ax,
            y_lim=y_lim,
            min_max=None,
            show_labels=False,  # Hide all labels, we'll add common ones later
            show_legend_and_title=False,
            save_plot=False,
        )
        
        maxp_ax.set_title("MaxPressure", fontsize=10)
        
        # Show only x-axis label for bottom plot
        if row != n_rows-1:
            maxp_ax.set_xticklabels([])
        maxp_ax.set_yticklabels([])
    
    # Add common legend in the middle section
    legend_ax = fig.add_subplot(gs[0:n_rows-1, 2], frameon=False)
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    
    # Organize legend items in the way specified: 2 columns, 8+1 rows
    # with specific grouping for the failure chance parameter
    if x_param == "failure chance":
        # For failure chance plots, the legend contains combinations of tpr and fpr
        params = ["true positive rate", "false positive rate"]
        values = {
            "true positive rate": [0.6, 0.7, 0.8, 0.9],
            "false positive rate": [0.0, 0.15, 0.3, 0.65]
        }
    elif x_param == "true positive rate":
        # For true positive rate plots, the legend contains combinations of fc and fpr
        params = ["failure chance", "false positive rate"]
        values = {
            "failure chance": [0.0, 0.05, 0.1, 0.15],
            "false positive rate": [0.0, 0.15, 0.3, 0.65]
        }
    else:  # x_param == "false positive rate"
        # For false positive rate plots, the legend contains combinations of fc and tpr
        params = ["failure chance", "true positive rate"]
        values = {
            "failure chance": [0.0, 0.05, 0.1, 0.15],
            "true positive rate": [0.6, 0.7, 0.8, 0.9]
        }
    
    # Find legend entries that correspond to the parameter combinations
    grouped_labels = []
    grouped_lines = []
    
    # Add title
    legend_title = f"Legend for {y_param_text} vs {x_param_text}"
    
    # First parameter group (first two values)
    for val1 in values[params[0]][:2]:
        for val2 in values[params[1]]:
            # Find matching label
            for label, line in zip(all_legend_labels, all_legend_lines):
                if f"{ABBREVIATIONS[params[0]]}={val1}" in label and f"{ABBREVIATIONS[params[1]]}={val2}" in label:
                    grouped_labels.append(label)
                    grouped_lines.append(line)
                    break
    
    # Add a spacer
    grouped_labels.append("")
    grouped_lines.append(None)
    
    # Second parameter group (second two values)
    for val1 in values[params[0]][2:]:
        for val2 in values[params[1]]:
            # Find matching label
            for label, line in zip(all_legend_labels, all_legend_lines):
                if f"{ABBREVIATIONS[params[0]]}={val1}" in label and f"{ABBREVIATIONS[params[1]]}={val2}" in label:
                    grouped_labels.append(label)
                    grouped_lines.append(line)
                    break
    
    # Add a spacer
    grouped_labels.append("")
    grouped_lines.append(None)
    
    # Add FixedTime label last
    fixedtime_label = next((label for label in all_legend_labels if "FixedTime" in label), None)
    if fixedtime_label:
        grouped_labels.append(fixedtime_label)
        grouped_lines.append(all_legend_lines[all_legend_labels.index(fixedtime_label)])
    
    # Create the legend with organized items
    legend = legend_ax.legend(
        grouped_lines, 
        grouped_labels,
        loc='center',
        fontsize=10,
        ncol=2,
        frameon=True,
        title=legend_title,
        title_fontsize=12
    )
    
    # Set the figure title at the top center
    fig.suptitle(
        f'{y_param_text} vs {x_param_text}\nSim: {sim}, Network: {network}',
        fontsize=16,
        y=0.98
    )
    
    # Add common axis labels
    fig.text(0.02, 0.5, y_param_text, va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.02, x_param_text, ha='center', fontsize=14)
    
    # Adjust layout
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])
    
    # Save the figure
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        filename = os.path.join(output_path, f'{ABBREVIATIONS[x_param_text]}_{ABBREVIATIONS[y_param_text]}_structured_grid.svg')
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
        
        # Also save as PNG for easier viewing
        png_filename = os.path.join(output_path, f'{ABBREVIATIONS[x_param_text]}_{ABBREVIATIONS[y_param_text]}_structured_grid.png')
        plt.savefig(png_filename, dpi=150)
        print(f"Saved plot to {png_filename}")
    
    return fig


def plot_data_for_all_metrics(
        files: dict[str, str],
        x_params: list[str],
        y_params: list[str],
        y_lims: list[tuple[float, float]],
        output_path: str = None,
        ):
    """
    Create structured grid plots for multiple metrics and parameters.
    
    Args:
        files (dict): Dictionary mapping agent labels to filepaths.
        x_params (list): List of parameters for the x-axis.
        y_params (list): List of performance metrics for the y-axis.
        y_lims (list): List of y-axis limits for each y_param.
        output_path (str): Path to save the output plots.
    """
    for x_param in x_params:
        for y_param, y_lim in zip(y_params, y_lims):
            create_structured_grid_plot(
                files=files,
                x_param=x_param,
                y_param=y_param,
                y_lim=y_lim,
                output_path=output_path
            )
            plt.close()


def get_exp_label(filepath: str) -> str:
    """
    From a given filepath, extract the training mode (disturbed/undisturbed) and the seed as the experiment label

    Args:
        filepath (str): The path to the experiment folder.

    Returns:
        str: The experiment label.
    """
    if "undisturbed" in filepath.lower():
        mode = "Undisturbed"
    elif "disturbed" in filepath.lower():
        mode = "Disturbed"
    elif "maxpressure" in filepath.lower():
        mode = "MaxPressure"
    else:
        raise ValueError(f"Unexpected value for mode in filepath: {filepath}")
    
    # Extract seed number
    str_seed = "Unknown"
    match = re.search(r'seed(\d+)', filepath)
    if match:
        str_seed = match.group(1)
    
    return f"{mode}, seed={str_seed}"


def main():
    from agent_comparison_plots import choose_experiments
    basepath = os.path.join("data", "output_data", "tsc")
    list_filepaths = choose_experiments()
    
    # Get experiment labels
    filepaths = {get_exp_label(filepath): filepath for filepath in list_filepaths}
    
    # Add MaxPressure
    filepaths["MaxPressure"] = os.path.join(
        basepath, 
        "sumo_maxpressure/sumo1x3/exp6_1_maxpressure/logger/2024_04_27-12_55_31_BRF.log"
    )
    
    # Create the output directory if it doesn't exist
    output_path = os.path.join("data", "output_data", "tsc", "structured_plots")
    os.makedirs(output_path, exist_ok=True)
    
    # Plot all metrics and parameters
    plot_data_for_all_metrics(
        files=filepaths,
        x_params=["failure chance", "true positive rate", "false positive rate"],
        y_params=["throughput", "travel_time"],
        y_lims=[(1400, 2900), (60, 300)],
        output_path=output_path
    )


if __name__ == "__main__":
    main()