
import os
from math import floor, ceil

import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import numpy as np
import hsluv
from pandas.plotting import parallel_coordinates

from data_reader import read_and_group_test_data, experiment_name, exp_config_from_path, ABBREVIATIONS, get_fixedtime_data
from test_data_plotting_combined import plot_averaged_data_with_range, get_exp_label

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
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as patches
    
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
            # Create figure
            fig = plt.figure(figsize=(24, 14))
            
            # Define grid with 5 columns: 2 for undisturbed, 1 for center, 2 for disturbed
            gs = GridSpec(6, 5, figure=fig, 
                          width_ratios=[1, 1, 1.5, 1, 1],  # Make the center column wider
                          height_ratios=[1, 1, 1, 1, 1, 1],
                          wspace=0.3, hspace=0.3)  # Increase spacing
            
            # Create axes for all the plots
            # Left side (undisturbed)
            undisturbed_axes = []
            for row in range(6):
                for col in range(2):
                    ax = fig.add_subplot(gs[row, col])
                    undisturbed_axes.append(ax)
                    # Turn off all tick labels initially
                    ax.tick_params(labelbottom=False, labelleft=False)
            
            # Middle column
            title_ax = fig.add_subplot(gs[0, 2])
            title_ax.axis('off')
            
            legend_ax = fig.add_subplot(gs[1:5, 2])
            legend_ax.axis('off')
            
            maxpressure_ax = fig.add_subplot(gs[5, 2])
            # Hide y-axis labels for MaxPressure plot
            maxpressure_ax.tick_params(labelleft=False)
            
            # Right side (disturbed)
            disturbed_axes = []
            for row in range(6):
                for col in range(3, 5):
                    ax = fig.add_subplot(gs[row, col])
                    disturbed_axes.append(ax)
                    # Turn off all tick labels initially
                    ax.tick_params(labelbottom=False, labelleft=False)
            
            # All legend items will be collected here
            all_legend_labels = []
            all_legend_lines = []
            
            # Plot data for undisturbed agents
            for i, (label, filepath) in enumerate(undisturbed_items):
                if i >= len(undisturbed_axes):
                    print(f"Warning: Too many undisturbed agents, skipping {label}")
                    continue
                
                ax = undisturbed_axes[i]
                exp_path = filepath.strip(os.path.basename(filepath))[:-len("logger/")]
                sample_exp_path = exp_path  # Save for later use
                data = read_and_group_test_data(filepath)
                
                # For the first plot, save all legend items
                if i == 0:
                    labels, lines = plot_averaged_data_with_range(
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
                    all_legend_labels = labels
                    all_legend_lines = lines
                else:
                    # For other plots, don't save legend items
                    plot_averaged_data_with_range(
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
                
                # Extract seed from label
                seed = label.split("seed=")[1] if "seed=" in label else ""
                ax.set_title(f"seed={seed}", fontsize=10)
                
                # Show tick labels only on bottom row and leftmost column
                row = i // 2
                col = i % 2
                
                if row == 5:  # Bottom row
                    ax.tick_params(labelbottom=True)
                if col == 0:  # Leftmost column
                    ax.tick_params(labelleft=True)
            
            # Plot data for MaxPressure
            for label, filepath in maxpressure_file.items():
                exp_path = filepath.strip(os.path.basename(filepath))[:-len("logger/")]
                data = read_and_group_test_data(filepath)
                
                plot_averaged_data_with_range(
                    data,
                    x_param,
                    y_param,
                    exp_path=exp_path,
                    ax=maxpressure_ax,
                    y_lim=y_lim,
                    show_labels=False,
                    show_legend_and_title=False,
                    save_plot=False,
                    min_max=None,
                )
                
                maxpressure_ax.set_title("MaxPressure", fontsize=10)
                maxpressure_ax.tick_params(labelbottom=True, labelleft=False)
            
            # Plot data for disturbed agents
            for i, (label, filepath) in enumerate(disturbed_items):
                if i >= len(disturbed_axes):
                    print(f"Warning: Too many disturbed agents, skipping {label}")
                    continue
                
                ax = disturbed_axes[i]
                exp_path = filepath.strip(os.path.basename(filepath))[:-len("logger/")]
                data = read_and_group_test_data(filepath)
                
                plot_averaged_data_with_range(
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
                
                # Extract seed from label
                seed = label.split("seed=")[1] if "seed=" in label else ""
                ax.set_title(f"seed={seed}", fontsize=10)
                
                # Show tick labels only on bottom row and rightmost column
                row = i // 2
                col = i % 2
                
                if row == 5:  # Bottom row
                    ax.tick_params(labelbottom=True)
                if col == 1:  # Rightmost column
                    ax.tick_params(labelright=True)
                    ax.yaxis.set_label_position("right")
            
                            # Create reorganized legend with 2 columns
            if all_legend_labels and all_legend_lines:
                # Find FixedTime label
                fixedtime_idx = None
                for i, label in enumerate(all_legend_labels):
                    if "FixedTime" in label:
                        fixedtime_idx = i
                        break
                
                # Extract all label information
                legend_map = {}
                for i, label in enumerate(all_legend_labels):
                    if i == fixedtime_idx:
                        continue  # Skip FixedTime
                    
                    parts = label.split(", ")
                    if len(parts) == 2:
                        fc_part = parts[0].split("=")
                        fpr_part = parts[1].split("=")
                        
                        if len(fc_part) == 2 and len(fpr_part) == 2:
                            fc = float(fc_part[1])
                            fpr = float(fpr_part[1])
                            legend_map[(fc, fpr)] = (label, all_legend_lines[i])
                
                # Define the exact order we want based on the image
                # The order should be:
                # Column 1: [fc=0.0, fpr=0.65], [fc=0.1, fpr=0.65], [fc=0.0, fpr=0.3], [fc=0.1, fpr=0.3], etc.
                # Column 2: [fc=0.05, fpr=0.65], [fc=0.15, fpr=0.65], [fc=0.05, fpr=0.3], [fc=0.15, fpr=0.3], etc.
                
                fc_order = [0.0, 0.1, 0.05, 0.15] # First two for left column, last two for right column
                fpr_order = [0.65, 0.3, 0.15, 0.0] # Brightest to darkest
                
                # Build new legend layout
                new_labels = []
                new_lines = []
                
                # Loop through fpr values (controls brightness)
                for fpr in fpr_order:
                    # Left column: fc=0.0
                    if (0.0, fpr) in legend_map:
                        new_labels.append(legend_map[(0.0, fpr)][0])
                        new_lines.append(legend_map[(0.0, fpr)][1])
                    else:
                        new_labels.append("")
                        new_lines.append(plt.Line2D([], [], alpha=0))
                    
                    # Right column: fc=0.05
                    if (0.05, fpr) in legend_map:
                        new_labels.append(legend_map[(0.05, fpr)][0])
                        new_lines.append(legend_map[(0.05, fpr)][1])
                    else:
                        new_labels.append("")
                        new_lines.append(plt.Line2D([], [], alpha=0))
                
                # Spacing row
                new_labels.append("")
                new_labels.append("")
                new_lines.append(plt.Line2D([], [], alpha=0))
                new_lines.append(plt.Line2D([], [], alpha=0))
                
                # Loop through fpr values again for the second group
                for fpr in fpr_order:
                    # Left column: fc=0.1
                    if (0.1, fpr) in legend_map:
                        new_labels.append(legend_map[(0.1, fpr)][0])
                        new_lines.append(legend_map[(0.1, fpr)][1])
                    else:
                        new_labels.append("")
                        new_lines.append(plt.Line2D([], [], alpha=0))
                    
                    # Right column: fc=0.15
                    if (0.15, fpr) in legend_map:
                        new_labels.append(legend_map[(0.15, fpr)][0])
                        new_lines.append(legend_map[(0.15, fpr)][1])
                    else:
                        new_labels.append("")
                        new_lines.append(plt.Line2D([], [], alpha=0))
                
                # Add FixedTime in the correct position (row 9, column 1)
                if fixedtime_idx is not None:
                    # Add empty line before FixedTime
                    new_labels.append("")
                    new_labels.append("")
                    new_lines.append(plt.Line2D([], [], alpha=0))
                    new_lines.append(plt.Line2D([], [], alpha=0))
                    
                    # Add FixedTime in the left column
                    new_labels.append(all_legend_labels[fixedtime_idx])
                    # Empty slot in right column
                    new_labels.append("")
                    new_lines.append(all_legend_lines[fixedtime_idx])
                    new_lines.append(plt.Line2D([], [], alpha=0))
                
                # Create the legend
                legend = legend_ax.legend(
                    new_lines, new_labels,
                    loc='center',
                    ncol=2,
                    fontsize=10,
                    frameon=True,
                    columnspacing=1.5,
                    handletextpad=1.0,
                    labelspacing=0.8
                )
            
            # Add title and global labels
            x_param_text = x_param.replace('_', ' ')
            y_param_text = y_param.replace('_', ' ')
            
            if sample_exp_path:
                sim, method, network, exp_name = exp_config_from_path(sample_exp_path, convert_network=True)
                subtitle = f"Sim: {sim}, Network: {network}"
                
                title_ax.text(0.5, 0.8, f'{y_param_text} vs {x_param_text}',
                             fontsize=16, ha='center', va='center')
                title_ax.text(0.5, 0.6, subtitle,
                             fontsize=12, ha='center', va='center')
            
            # Add section headers - position closer to the plots
            fig.text(0.25, 0.96, "Undisturbed", fontsize=14, ha='center')
            fig.text(0.75, 0.96, "Disturbed", fontsize=14, ha='center')
            
            # Add global axis labels - position closer to the plots
            fig.text(0.025, 0.5, y_param_text, fontsize=14, rotation=90, ha='center', va='center')
            fig.text(0.5, 0.02, x_param_text, fontsize=14, ha='center', va='center')
            
            # Create larger rectangular outlines around agent groups with more padding
            if undisturbed_axes:
                # Get the position of the first and last subplot
                first_pos = undisturbed_axes[0].get_position()
                last_pos = undisturbed_axes[-1].get_position()
                
                # Add extra padding to avoid intersecting with labels
                padding = 0.03
                undisturbed_rect = patches.Rectangle(
                    (first_pos.x0 - padding, last_pos.y0 - padding),
                    (undisturbed_axes[1].get_position().x1 - first_pos.x0) + 2*padding,
                    (first_pos.y1 - last_pos.y0) + 2*padding,
                    linewidth=1, edgecolor='#cccccc', facecolor='none',
                    transform=fig.transFigure, zorder=-1  # Set lower zorder so it's behind the plots
                )
                fig.patches.append(undisturbed_rect)
            
            if disturbed_axes:
                # Get the position of the first and last subplot
                first_pos = disturbed_axes[0].get_position()
                last_pos = disturbed_axes[-1].get_position()
                
                # Add extra padding to avoid intersecting with labels
                padding = 0.03
                disturbed_rect = patches.Rectangle(
                    (first_pos.x0 - padding, last_pos.y0 - padding),
                    (disturbed_axes[1].get_position().x1 - first_pos.x0) + 2*padding,
                    (first_pos.y1 - last_pos.y0) + 2*padding,
                    linewidth=1, edgecolor='#cccccc', facecolor='none',
                    transform=fig.transFigure, zorder=-1  # Set lower zorder so it's behind the plots
                )
                fig.patches.append(disturbed_rect)
            
            # Save the plot
            if output_path:
                os.makedirs(output_path, exist_ok=True)
                filename = os.path.join(output_path, f'{ABBREVIATIONS[x_param_text]}_{ABBREVIATIONS[y_param_text]}_new_layout.svg')
                fig.savefig(filename)
                print(f"Saved plot to {filename}")
            
            plt.close(fig)
                
def main():
    from agent_comparison_plots import choose_experiments
    basepath = os.path.join("data", "output_data", "tsc") # if necessary, add `os.path.dirname("."), ` to the front of paths
    list_filepaths = choose_experiments()
    filepaths: dict[str, str] = {get_exp_label(filepath): filepath for filepath in list_filepaths}
    filepaths["MaxPressure"] = os.path.join(basepath, "sumo_maxpressure/sumo1x3/exp6_1_maxpressure/logger/2024_04_27-12_55_31_BRF.log")
    
    # Replace the existing plot_data_for_all_agents call with the new function
    plot_data_for_all_agents(
        files=filepaths,
        x_params=["failure chance", "true positive rate", "false positive rate"],
        y_params=["throughput", "travel_time"],
        y_lims=[(1400, 2900), (60, 300)],
        output_path=os.path.join("data", "output_data", "tsc", "structured_plots")
    )

if __name__ == '__main__':
    main()