"""
This module provides tools to create grid-based comparison plots for agent performance 
across different noise settings.

Authors: Sebastian Jost & Claude (16.03.2025)
"""

import os
from math import floor, ceil

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np

from data_reader import read_and_group_test_data, experiment_name, exp_config_from_path, ABBREVIATIONS, get_fixedtime_data
from test_data_plotting_combined import plot_averaged_data_with_range, get_exp_label, parameters_to_color

def create_figure_layout(n_rows=6):
    """
    Create a figure with a grid layout for agent comparison plots.
    
    Args:
        n_rows (int): Number of rows for agent plots
        
    Returns:
        tuple: (fig, axs_dict) containing the figure and dictionary of axes
    """
    # Create figure
    fig = plt.figure(figsize=(24, 14))
    
    # Define grid with 5 columns: 2 for undisturbed, 1 for center, 2 for disturbed
    gs = GridSpec(n_rows, 5, figure=fig, 
                  width_ratios=[1, 1, 1.5, 1, 1],  # Make the center column wider
                  height_ratios=[1] * n_rows,
                  wspace=0.3, hspace=0.3)  # Increase spacing
    
    # Create axes for all the plots
    axs_dict = {
        'undisturbed': [],
        'disturbed': [],
        'title': fig.add_subplot(gs[0, 2]),
        'legend': fig.add_subplot(gs[1:4, 2]),
        'maxpressure': fig.add_subplot(gs[4:5, 2])
    }
    
    # Turn off axes for title and legend
    axs_dict['title'].axis('off')
    axs_dict['legend'].axis('off')
    
    # Hide y-axis labels for MaxPressure plot
    axs_dict['maxpressure'].tick_params(labelleft=False)
    
    # Left side (undisturbed)
    for row in range(n_rows):
        for col in range(2):
            ax = fig.add_subplot(gs[row, col])
            axs_dict['undisturbed'].append(ax)
            # Turn off all tick labels initially
            ax.tick_params(labelbottom=False, labelleft=False)
    
    # Right side (disturbed)
    for row in range(n_rows):
        for col in range(3, 5):
            ax = fig.add_subplot(gs[row, col])
            axs_dict['disturbed'].append(ax)
            # Turn off all tick labels initially
            ax.tick_params(labelbottom=False, labelleft=False)
    
    return fig, axs_dict


def configure_axes_labels(axs_list, side='left'):
    """
    Configure which axes show tick labels based on their position.
    
    Args:
        axs_list (list): List of axes to configure
        side (str): 'left' or 'right' to determine which side's y-axis to show
    """
    for i, ax in enumerate(axs_list):
        row = i // 2
        col = i % 2
        
        # Show x-axis labels only on bottom row
        if row == 5:  # Bottom row
            ax.tick_params(labelbottom=True)
            
        # Show y-axis labels based on side
        if side == 'left' and col == 0:  # Leftmost column
            ax.tick_params(labelleft=True)
        elif side == 'right' and col == 1:  # Rightmost column
            ax.tick_params(labelright=True)
            ax.yaxis.set_label_position("right")


def extract_seed_from_label(label):
    """Extract the seed number from an agent label."""
    return label.split("seed=")[1] if "seed=" in label else ""


def plot_agent_data(ax, filepath, x_param, y_param, y_lim):
    """
    Plot data for a single agent on the given axis.
    
    Args:
        ax (matplotlib.axes.Axes): The axis to plot on
        filepath (str): Path to the agent's data file
        x_param (str): X-axis parameter
        y_param (str): Y-axis parameter
        y_lim (tuple): Y-axis limits
        
    Returns:
        tuple: (legend_labels, legend_lines) if this is the first plot, else None
    """
    exp_path = filepath.strip(os.path.basename(filepath))[:-len("logger/")]
    data = read_and_group_test_data(filepath)
    
    return plot_averaged_data_with_range(
        data,
        x_param,
        y_param,
        exp_path=exp_path,
        ax=ax,
        y_lim=y_lim,
        show_labels=False,
        show_legend_and_title=False,
        save_plot=False,
        min_max=None,
    )


def reorganize_legend_items(legend_labels, legend_lines):
    """
    input structure:
    - 4 columns, 5 rows
    - one color per column
    - 4 colors of similar brightness per row
    - 5th row only for fixedtime

    output structure:
    - 2 columns, 9 rows
    - two colors per column
    - 2 colors of similar brightness per row
    - 9th row only for fixedtime

    Args:
        legend_labels (list): List of legend labels
        legend_lines (list): List of legend lines
    
    Returns:
        list[str]: New legend labels (reordered)
        list[Line2D]: New legend lines (reordered)
    """
    index_mapping = [
        0, 1, 2, 3,     # first color
        13, 14, 15, 16, # fourth color
        4,              # fixedtime
        5, 6, 7, 8,     # second color
        9, 10, 11, 12,  # third color
    ]

    
    new_labels = [legend_labels[index_mapping[i]] for i in range(len(legend_labels))]
    new_lines = [legend_lines[index_mapping[i]] for i in range(len(legend_lines))]

    
    return new_labels, new_lines


def add_section_outlines(fig, axs_dict):
    """
    Add rectangular outlines around the undisturbed and disturbed groups.
    
    Args:
        fig (matplotlib.figure.Figure): The figure
        axs_dict (dict): Dictionary of axes
    """
    # Add rectangular outlines with padding
    padding = 0.02  # Increased padding
    
    for color, axes in [('#2c2', axs_dict['undisturbed']), ('#58f', axs_dict['disturbed'])]:
        if not axes:
            continue
            
        # Get the position of the first and last subplot
        first_pos = axes[0].get_position()
        last_pos = axes[-1].get_position()
        right_edge = axes[1].get_position().x1 if len(axes) > 1 else first_pos.x1
        
        # Create rectangle with padding
        rect = patches.Rectangle(
            (first_pos.x0 - padding, last_pos.y0 - padding),
            (right_edge - first_pos.x0) + 2*padding,
            (first_pos.y1 - last_pos.y0) + 2*padding,
            linewidth=2,
            edgecolor=color,
            facecolor='none',
            transform=fig.transFigure,
            zorder=-1,  # Set lower zorder so it's behind the plots
        )
        fig.patches.append(rect)


def add_titles_and_labels(fig, axs_dict, x_param, y_param, exp_path=None):
    """
    Add titles, subtitles, and axis labels to the plot.
    
    Args:
        fig (matplotlib.figure.Figure): The figure
        axs_dict (dict): Dictionary of axes
        x_param (str): X-axis parameter
        y_param (str): Y-axis parameter
        exp_path (str): Path to the experiment folder
    """
    x_param_text = x_param.replace('_', ' ')
    y_param_text = y_param.replace('_', ' ')
    
    # Add title and subtitle
    if exp_path:
        sim, method, network, exp_name = exp_config_from_path(exp_path, convert_network=True)
        subtitle = f"Sim: {sim}, Network: {network}"
        
        axs_dict['title'].text(0.5, 0.8, f'{y_param_text} vs {x_param_text}',
                     fontsize=16, ha='center', va='center', fontweight='bold')
        axs_dict['title'].text(0.5, 0.6, subtitle,
                     fontsize=12, ha='center', va='center')
    
    # Add section headers - position closer to the plots
    fig.text(0.25, 0.93, "Undisturbed", fontsize=14, ha='center', color='#2c2', fontweight='bold')
    fig.text(0.75, 0.93, "Disturbed", fontsize=14, ha='center', color='#58f', fontweight='bold')
    
    # Add global axis labels - position closer to the plots
    fig.text(0.05, 0.5, y_param_text, fontsize=14, rotation=90, ha='center', va='center')
    fig.text(0.5, 0.05, x_param_text, fontsize=14, ha='center', va='center')


def plot_data_for_all_agents(
        files: dict[str, str],
        x_params: list[str],
        y_params: list[str],
        y_lims: list[tuple[float, float]],
        output_path: str = None,
        ):
    """
    Create a complex grid layout with undisturbed agents on the left, disturbed agents on the right,
    and a central column with title, legend, and MaxPressure plot.
    
    Args:
        files (dict[str, str]): Dictionary mapping agent labels to file paths.
        x_params (list[str]): List of x-axis parameters to plot.
        y_params (list[str]): List of y-axis parameters to plot.
        y_lims (list[tuple[float, float]]): List of y-axis limits for each y_param.
        output_path (str, optional): Path to save the plots. Defaults to None.
    """
    # Separate agents by type
    undisturbed_files = {k: v for k, v in files.items() if "Undisturbed" in k}
    disturbed_files = {k: v for k, v in files.items() if "Disturbed" in k}
    maxpressure_file = {k: v for k, v in files.items() if "MaxPressure" in k}
    
    # Get a list of files for each type to ensure consistent order
    undisturbed_items = list(undisturbed_files.items())
    disturbed_items = list(disturbed_files.items())
    
    # Process each parameter combination
    for y_param, y_lim in zip(y_params, y_lims):
        for x_param in x_params:
            # Create figure and layout
            fig, axs_dict = create_figure_layout()
            
            # Track first experiment path for metadata
            sample_exp_path = None
            legend_items = None
            
            # Plot data for undisturbed agents
            for i, (label, filepath) in enumerate(undisturbed_items):
                if i >= len(axs_dict['undisturbed']):
                    print(f"Warning: Too many undisturbed agents, skipping {label}")
                    continue
                
                ax = axs_dict['undisturbed'][i]
                exp_path = filepath.strip(os.path.basename(filepath))[:-len("logger/")]
                sample_exp_path = exp_path if sample_exp_path is None else sample_exp_path
                
                # Plot data
                ordered_labels, ordered_lines = plot_agent_data(ax, filepath, x_param, y_param, y_lim)
                
                # Save legend items from the first plot
                legend_items = ordered_labels, ordered_lines
                
                # Add seed label
                seed = extract_seed_from_label(label)
                ax.set_title(f"seed={seed}", fontsize=10)
            
            # Configure axes labels for undisturbed plots
            configure_axes_labels(axs_dict['undisturbed'], side='left')
            
            # Plot data for MaxPressure
            for label, filepath in maxpressure_file.items():
                plot_agent_data(axs_dict['maxpressure'], filepath, x_param, y_param, y_lim)
                axs_dict['maxpressure'].set_title("MaxPressure", fontsize=10)
                axs_dict['maxpressure'].tick_params(labelbottom=True, labelleft=False)
            
            # Plot data for disturbed agents
            for i, (label, filepath) in enumerate(disturbed_items):
                if i >= len(axs_dict['disturbed']):
                    print(f"Warning: Too many disturbed agents, skipping {label}")
                    continue
                
                ax = axs_dict['disturbed'][i]
                
                # Plot data
                plot_agent_data(ax, filepath, x_param, y_param, y_lim)
                
                # Add seed label
                seed = extract_seed_from_label(label)
                ax.set_title(f"seed={seed}", fontsize=10)
            
            # Configure axes labels for disturbed plots
            configure_axes_labels(axs_dict['disturbed'], side='right')
            
            # Create legend if we have legend items
            if legend_items:
                # Unpack legend items
                legend_labels, legend_lines = legend_items
                
                # Create properly ordered legend items
                new_labels, new_lines = reorganize_legend_items(legend_labels, legend_lines)
                
                # Create the legend
                legend = axs_dict['legend'].legend(
                    new_lines, new_labels,
                    loc='center',
                    ncol=2,
                    fontsize=10,
                    frameon=True,
                    columnspacing=1.5,
                    handletextpad=1.0,
                    labelspacing=0.8
                )
            
            # Add title and labels
            add_titles_and_labels(fig, axs_dict, x_param, y_param, sample_exp_path)
            
            # Add outlines around plot groups
            add_section_outlines(fig, axs_dict)
            
            # Save the plot
            if output_path:
                x_param_text: str = x_param.replace("_", " ")
                y_param_text: str = y_param.replace("_", " ")
                os.makedirs(output_path, exist_ok=True)
                filename = os.path.join(output_path, f'{ABBREVIATIONS[x_param_text]}_{ABBREVIATIONS[y_param_text]}_grid.svg')
                fig.savefig(filename)
                print(f"Saved plot to {filename}")
            
            plt.close(fig)


def main():
    from agent_comparison_plots import choose_experiments
    basepath = os.path.join("data", "output_data", "tsc") 
    list_filepaths = choose_experiments()
    filepaths: dict[str, str] = {get_exp_label(filepath): filepath for filepath in list_filepaths}
    # filepaths["MaxPressure"] = os.path.join(basepath, "sumo_maxpressure/sumo1x3/exp6_1_maxpressure/logger/2024_04_27-12_55_31_BRF.log") # old maxpressure test (original dataset)
    filepaths["MaxPressure"] = os.path.join(basepath, "sumo_maxpressure/sumo1x3/exp_29072025_maxpressure/logger/2025_07_29-18_23_17_BRF.log")
    
    plot_data_for_all_agents(
        files=filepaths,
        x_params=["failure chance", "true positive rate", "false positive rate"],
        y_params=["throughput", "travel_time"],
        y_lims=[(900, 2900), (60, 450)],
        output_path=os.path.join("data", "output_data", "tsc", "structured_plots")
    )

if __name__ == '__main__':
    main()