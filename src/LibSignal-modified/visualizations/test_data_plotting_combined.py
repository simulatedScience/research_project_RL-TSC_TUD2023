"""
This module provides tools to summarize and plot test data of a single agent for many noise settings.

The user can select multiple plots, which are then averaged based on the experiment name (grouped by disturbed, undisturbed, maxPressure, FixedTime)

In the plots, each noise parameter ist represented by one color component (Hue, Saturation, brightness) and the x-axis is used for the noise parameter that is not represented by a color component. This allows easy visualization of the effect of different noise settings on the performance metrics.

Authors: Sebastian Jost & GPT-4 (24.10.2023)
"""

import os
from math import floor, ceil
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import numpy as np
import hsluv
# pandas is no longer needed in the provided snippet if parallel_coordinates isn't used elsewhere
# from pandas.plotting import parallel_coordinates

# Assuming these are correctly defined in data_reader or globally
from data_reader import read_and_group_test_data, experiment_name, exp_config_from_path, ABBREVIATIONS, get_fixedtime_data

def plot_averaged_data_with_range(
        segregated_data: dict,
        x_param: str,
        y_param: str,
        exp_path: str, # Still needed for experiment config, use one representative path
        ax: plt.Axes = None,
        y_lim: tuple = None,
        min_max: bool = None, # Changed default to None to distinguish between stddev and no range
        show_labels: bool = True,
        show_legend_and_title: bool = True,
        save_plot: bool = True,
        ) -> tuple[list[str], list[plt.Line2D]]:
    """
    Plot the averaged metric for each group along with its range.

    Args:
        segregated_data (dict): Dictionary grouping data by noise settings.
                                (Assumes this data might already be pre-averaged across seeds)
        x_param (str): The noise setting for the x-axis ("failure chance", "true positive rate", or "false positive rate").
        y_param (str): The performance metric for the y-axis (e.g., "throughput", "delay").
        exp_path (str): Path to a representative experiment folder (for config details).
        ax (plt.Axes): The axis to plot on (default is None -> create a new figure).
        y_lim (tuple): The y-axis limits (default is None -> fit to data).
        min_max (bool | None): Whether to use the minimum and maximum values for the range (True),
                               the standard deviation (False), or no range (None).
        show_labels (bool): Whether to show axis labels (default is True).
        show_legend_and_title (bool): Whether to show the legend and plot title (default is True).
        save_plot (bool): Whether to save the plot as a .svg file (default is True).

    Returns:
        (list[str]): List of labels for the legend.
        (list[plt.Line2D]): List of lines for the legend.
    """
    # This function receives data that might be averaged across seeds OR just from one run (like MaxPressure)
    # compute first-level averages (if not already done), segregate, plot lines.

    # If the input `segregated_data` is raw (dict keyed by noise tuple), compute averages.
    # If it's already averaged (list of dicts), use it directly.
    # For simplicity, let's assume `compute_averages` handles both or we ensure the correct format is passed.
    # **Modification**: Check input type. If it's the raw dict from read_and_group, compute averages.
    # If it's the list structure from `aggregate_category_data`, skip `compute_averages`.
    if isinstance(segregated_data, dict) and \
       len(segregated_data) > 0 and \
       isinstance(list(segregated_data.keys())[0], tuple): # Heuristic check for raw grouped data
        average_data = compute_averages(segregated_data)
    elif isinstance(segregated_data, list): # Assumes it's already averaged data structure
        average_data = segregated_data
    else:
        # Handle empty data or unexpected format
        print(f"Warning: Unexpected or empty data format for plotting {y_param} vs {x_param}. Skipping.")
        return [], []

    segregated_by_plot_lines: dict[tuple[float, float], list[dict]] = segregate_data_by_params(average_data, x_param)
    segregated_by_plot_lines = dict(sorted(segregated_by_plot_lines.items()))

    if ax is None:
        fig = plt.figure(figsize=(8.8 / 1.5, 10 / 1.5)) # Adjust size if needed
        fig.tight_layout()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.85) # Adjust margins
        print("Added subplot (single plot mode)")

    legend_mapping = {}
    legend_labels = []

    noise_ranges = {
        "fc": (0, 0.2),
        "tpr": (0.6, 1.),
        "fpr": (0, 0.65),
    }
    # Invert TPR range for color mapping consistency if needed, but color logic below uses index now
    # noise_ranges["tpr"] = (1-noise_ranges['tpr'][1], 1-noise_ranges['tpr'][0])
    hsluv_ranges = [
        (0., 240.),   # Hue range
        (0., 100.),  # Saturation range (full range allowed)
        (35., 85.),   # Lightness range (avoid too dark/light)
    ]

    num_lines = len(segregated_by_plot_lines)
    num_cols_legend = 4 # assume 4 levels for the other params for coloring/legend layout

    for idx, (key, group) in enumerate(segregated_by_plot_lines.items()):
        if not group: continue # Skip empty groups

        x_values = [run[x_param] for run in group]
        avg_values = [run[y_param]['average'] for run in group]

        if min_max is True:
            min_values = [run[y_param]['min'] for run in group]
            max_values = [run[y_param]['max'] for run in group]
        elif min_max is False: # use standard deviation
            min_values = [run[y_param]['average'] - run[y_param]['std'] for run in group]
            max_values = [run[y_param]['average'] + run[y_param]['std'] for run in group]
        # else: min_max is None, don't plot range

        # Sort data by x values
        sorted_indices = sorted(range(len(x_values)), key=lambda k: x_values[k])
        x_values = [x_values[i] for i in sorted_indices]
        avg_values = [avg_values[i] for i in sorted_indices]
        if min_max is not None:
            min_values = [min_values[i] for i in sorted_indices]
            max_values = [max_values[i] for i in sorted_indices]

        other_params = [ABBREVIATIONS[param] for param in ['failure chance', 'true positive rate', 'false positive rate'] if param != x_param]
        label = f"{other_params[0]}={key[0]}, {other_params[1]}={key[1]}"

        # --- Color Calculation based on index/grid position ---
        # This assumes a fixed grid (e.g., 4x4 for the other two params)
        # Ensure idx maps correctly if the number of lines isn't exactly 16
        row = idx // num_cols_legend
        col = idx % num_cols_legend
        # Map row/col to HSLuv components (example mapping, adjust as needed)
        h = (col / (num_cols_legend -1 )) * 240 if num_cols_legend > 1 else 0 # Hue based on one param variation
        l = 40 + (row / (num_lines / num_cols_legend -1)) * 40 if (num_lines / num_cols_legend) > 1 else 60 # Lightness based on other
        s = 80 # Fixed saturation
        color = hsluv.hsluv_to_rgb([h, s, l])
        # --- End Color Calculation ---

        if min_max is not None:
            ax.fill_between(x_values, min_values, max_values, color=color, alpha=0.2)
        line, = ax.plot(x_values, avg_values, 'o-', label=label, color=color)
        legend_mapping[label] = line
        legend_labels.append(label)

    # Add reference line for fixedtime if data available
    try:
        fixedtime_data = get_fixedtime_data()
        if y_param in fixedtime_data:
            fixedtime_label = 'FixedTime 30s'
            fixedtime_line = ax.axhline(y=fixedtime_data[y_param], color='black', linestyle='--', alpha=0.5, label=fixedtime_label)
            legend_mapping[fixedtime_label] = fixedtime_line
            legend_labels.append(fixedtime_label)
        else:
            print(f"Warning: Metric '{y_param}' not found in fixedtime_data.")
    except Exception as e:
        print(f"Warning: Could not get or plot fixedtime data. Error: {e}")


    x_param_text = x_param.replace('_', ' ')
    y_param_text = y_param.replace('_', ' ')

    sim, method, network, exp_name = exp_config_from_path(exp_path, convert_network=True)
    exp_subtitle = f"Method: {method}\nExp: {exp_name}" # This might be less relevant now

    # Reordering the legend entries for consistency (assuming 4 columns)
    reordered_labels = legend_labels # Start with current order
    if num_lines == 16: # Only reorder if we have the expected 4x4 grid
        reordered_labels = [legend_labels[i::num_cols_legend] for i in range(num_cols_legend)]
        reordered_labels = [label for sublist in reordered_labels for label in sublist]
        # Add fixed time back if it was there
        if fixedtime_label in legend_mapping:
            if fixedtime_label not in reordered_labels:
                 reordered_labels.append(fixedtime_label)
        else: # Remove fixed time if it wasn't plotted
            reordered_labels = [lbl for lbl in reordered_labels if lbl != fixedtime_label]

    # Ensure reordered_lines matches reordered_labels
    reordered_lines = [legend_mapping[label] for label in reordered_labels if label in legend_mapping]
    # Update labels list to match lines returned
    reordered_labels = [label for label in reordered_labels if label in legend_mapping]


    if show_labels:
        ax.set_xlabel(x_param_text)
        ax.set_ylabel(y_param_text)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.grid(color="#dddddd")
    if show_legend_and_title:
        # Title might be set per subplot by the calling function now
        # ax.set_title(f'{y_param_text} vs {x_param_text}\n{exp_subtitle}')
        # Legend is likely handled globally by the calling function
        # ax.legend(reordered_lines, reordered_labels, loc='best', ncol=num_cols_legend)
        pass # Title and legend handled by plot_comparison_figures

    if save_plot:
        os.makedirs(os.path.join(exp_path, 'plots'), exist_ok=True)
        # Filename needs context from the calling function (e.g., category)
        filename = os.path.join(exp_path, 'plots', f'{ABBREVIATIONS.get(x_param_text, x_param_text)}_{ABBREVIATIONS.get(y_param_text, y_param_text)}_single.svg')
        plt.savefig(filename)
        print(f"Saved single plot to {filename}")
        plt.close(fig) # Close the figure if it was created here
        plt.clf() # Clear figure state

    # Return reordered labels/lines for potential global legend
    return reordered_labels, reordered_lines


# --- compute_averages remains the same ---
# (It averages over test runs for a single training seed/file)
def compute_averages(grouped_data: dict) -> list:
    """
    Compute average and range for each metric within each group (noise setting).

    Args:
    - grouped_data (dict): Dictionary grouping data by noise settings tuple (fc, tpr, fpr).
                           Values are dicts mapping metric names to lists of values from test runs.

    Returns:
    - list: A list of dictionaries, each containing settings and aggregated metrics (avg, min, max, std).
            Example element: {'failure chance': 0.1, ..., 'throughput': {'average': ..., 'min': ..., ...}}
    """
    averaged_data = []

    for key, metrics in grouped_data.items():
        if not isinstance(key, tuple) or len(key) != 3:
            print(f"Warning: Skipping unexpected key in compute_averages: {key}")
            continue

        averaged_run = {
            'failure chance': key[0],
            'true positive rate': key[1],
            'false positive rate': key[2]
        }
        for metric, values in metrics.items():
            if not values: # Handle case where a metric might be missing for a specific noise setting
                avg_value, min_value, max_value, std_value = np.nan, np.nan, np.nan, np.nan
            else:
                values = [v for v in values if v is not None and not np.isnan(v)] # Clean Nones/NaNs
                if not values:
                     avg_value, min_value, max_value, std_value = np.nan, np.nan, np.nan, np.nan
                else:
                    avg_value = sum(values) / len(values)
                    min_value = min(values)
                    max_value = max(values)
                    # Calculate std dev, handle single value case
                    std_value = np.std(values) if len(values) > 1 else 0.0 # Use 0 std for single point

            averaged_run[metric] = {
                'average': avg_value,
                'min': min_value,
                'max': max_value,
                'std': std_value,
            }

        averaged_data.append(averaged_run)

    return averaged_data


# --- segregate_data_by_params remains the same ---
# (It groups data for plotting lines based on the x_param)
def segregate_data_by_params(data: list, x_param: str) -> dict:
    """
    Segregate averaged data based on noise settings other than the chosen x_param.

    Args:
    - data (list): A list of dictionaries (output from compute_averages or aggregate_category_data).
    - x_param (str): The noise setting chosen for the x-axis.

    Returns:
    - dict: A dictionary grouping data points that belong to the same plot line.
            Keys are tuples of the values of the *other two* noise parameters.
            Values are lists of the original dictionaries from the input `data`.
    """
    other_params = [param for param in ['failure chance', 'true positive rate', 'false positive rate'] if param != x_param]
    if len(other_params) != 2:
        raise ValueError(f"Invalid x_param '{x_param}', could not determine other two parameters.")

    segregated_data = {}
    for run in data:
        try:
            key = (run[other_params[0]], run[other_params[1]])
            if key not in segregated_data:
                segregated_data[key] = []
            segregated_data[key].append(run)
        except KeyError as e:
            print(f"Warning: Missing parameter {e} in run data: {run}. Skipping this run for segregation.")
            continue

    return segregated_data

# --- NEW Function: Aggregate results across multiple files/seeds ---
def aggregate_category_data(filepaths: list[str], metrics_to_aggregate: list[str]) -> list:
    """
    Reads data from multiple filepaths, computes averages for each,
    and then aggregates these averages across all files.

    Args:
        filepaths (list[str]): List of paths to log files for this category.
        metrics_to_aggregate (list[str]): List of metric names (like 'throughput') to process.

    Returns:
        list: A list of dictionaries, mirroring the structure from compute_averages,
              but with values ('average', 'min', 'max', 'std') being the mean of those
              values across the input files for each noise setting. Returns empty list if no files.
    """
    if not filepaths:
        return []

    # Stores aggregated values PER noise setting across files
    # Structure: noise_tuple -> metric_name -> stat_name -> list_of_values_from_files
    aggregated_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    num_files_processed_per_noise = defaultdict(int)

    for filepath in filepaths:
        try:
            # 1. Read raw data for one file
            raw_grouped_data = read_and_group_test_data(filepath)
            if not raw_grouped_data:
                print(f"Warning: No data read from {filepath}. Skipping.")
                continue

            # 2. Compute averages over test runs for this file
            single_file_averages = compute_averages(raw_grouped_data)

            # 3. Add the computed stats to the overall aggregation dict
            for avg_run in single_file_averages:
                noise_key = (
                    avg_run['failure chance'],
                    avg_run['true positive rate'],
                    avg_run['false positive rate']
                )
                num_files_processed_per_noise[noise_key] += 1
                for metric in metrics_to_aggregate:
                    if metric in avg_run:
                        for stat in ['average', 'min', 'max', 'std']:
                            # Ensure value is not NaN before appending
                            value = avg_run[metric].get(stat, np.nan)
                            if value is not None and not np.isnan(value):
                                aggregated_stats[noise_key][metric][stat].append(value)
                    # else: Metric not present in this run's data, will be handled later

        except Exception as e:
            print(f"Error processing file {filepath}: {e}. Skipping.")
            continue

    # 4. Compute the final average of the aggregated stats
    final_averaged_data = []
    for noise_key, metrics_data in aggregated_stats.items():
        final_run = {
            'failure chance': noise_key[0],
            'true positive rate': noise_key[1],
            'false positive rate': noise_key[2]
        }
        # num_files = num_files_processed_per_noise[noise_key] # Use this if needed

        for metric in metrics_to_aggregate:
            final_run[metric] = {}
            if metric in metrics_data:
                for stat in ['average', 'min', 'max', 'std']:
                    values = metrics_data[metric].get(stat, [])
                    if values:
                        # Calculate mean of the collected stats (mean of averages, mean of mins, etc.)
                        final_run[metric][stat] = np.mean(values)
                    else:
                        # If no valid values were collected (e.g., all NaN or metric missing)
                        final_run[metric][stat] = np.nan
            else:
                # Metric was missing entirely for this noise setting across files
                 final_run[metric] = {'average': np.nan, 'min': np.nan, 'max': np.nan, 'std': np.nan}

        final_averaged_data.append(final_run)

    return final_averaged_data


# --- REVISED Function: plot_data_for_all_agents is now plot_comparison_figures ---
def plot_comparison_figures(
        files: dict[str, str],
        x_params: list[str],
        y_params: list[str],
        y_lims: list[tuple[float, float]],
        output_path: str = None,
        use_min_max_range: bool = None # Use min/max (True), std dev (False), or no range (None)
        ):
    """
    Generates comparison plots with three subplots: Undisturbed, MaxPressure, Disturbed.
    Averages results for Undisturbed and Disturbed categories if multiple files are provided.

    Args:
        files (dict[str, str]): Dict mapping labels to filepaths. Labels MUST contain
                               'Undisturbed', 'Disturbed', or 'MaxPressure'.
        x_params (list[str]): List of parameters for the x-axis.
        y_params (list[str]): List of metrics for the y-axis.
        y_lims (list[tuple[float, float]]): List of y-axis limits corresponding to y_params.
        output_path (str): Directory to save the plots.
        use_min_max_range (bool | None): How to display the range on plots.
    """
    # 1. Group files by category
    undisturbed_files = []
    disturbed_files = []
    maxpressure_files = [] # Should ideally be only one
    misc_files = []

    for label, filepath in files.items():
        if "Undisturbed" in label:
            undisturbed_files.append(filepath)
        elif "Disturbed" in label:
            disturbed_files.append(filepath)
        elif "MaxPressure" in label:
            maxpressure_files.append(filepath)
        else:
            misc_files.append(filepath)
            print(f"Warning: File with label '{label}' doesn't fit standard categories. Ignoring.")

    if not maxpressure_files:
        print("Warning: No 'MaxPressure' file found. MaxPressure plot will be empty.")
        # Decide how to handle - skip MP plot, plot empty, raise error? For now, allow empty.
        maxpressure_file = None
    elif len(maxpressure_files) > 1:
        print(f"Warning: Multiple 'MaxPressure' files found. Using the first one: {maxpressure_files[0]}")
        maxpressure_file = maxpressure_files[0]
    else:
        maxpressure_file = maxpressure_files[0]

    print(f"Found {len(undisturbed_files)} Undisturbed files.")
    print(f"Found {len(disturbed_files)} Disturbed files.")
    if maxpressure_file:
        print(f"Found MaxPressure file: {maxpressure_file}")

    # Get a representative exp_path (e.g., for simulation config) - use the first available file
    first_file = next(iter(files.values()), None)
    if not first_file:
        print("Error: No input files provided.")
        return
    rep_exp_path = os.path.dirname(os.path.dirname(first_file)) # Go up from logger/ to exp folder


    # --- Loop through desired plot configurations ---
    for y_param, y_lim in zip(y_params, y_lims):
        for x_param in x_params:
            print(f"\nGenerating plot for Y={y_param}, X={x_param}")
            fig, axs = plt.subplots(1, 3, sharey=True, figsize=(13, 6.5)) # 3 fixed subplots
            fig.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.79, wspace=0.05) # Adjust right for legend

            plot_titles = ["Undisturbed", "MaxPressure", "Disturbed"]
            category_files = [undisturbed_files, [maxpressure_file] if maxpressure_file else [], disturbed_files]

            legend_labels, legend_lines = None, None # To store legend info from the first valid plot

            # --- Process and Plot each category ---
            for i, (title, file_list) in enumerate(zip(plot_titles, category_files)):
                ax = axs[i]
                ax.set_title(title)
                category_data = None

                if not file_list:
                    print(f"No files for category '{title}'. Skipping plot.")
                    ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='grey')
                    continue

                try:
                    if title in ["Undisturbed", "Disturbed"]:
                        # Aggregate over multiple seeds
                        category_data = aggregate_category_data(file_list, y_params + [p for p in ABBREVIATIONS.values()]) # Ensure all needed metrics are aggregated
                        print(f"Aggregated data for {title} from {len(file_list)} files.")
                    elif title == "MaxPressure":
                        # Just read and compute averages for the single file
                        raw_data = read_and_group_test_data(file_list[0])
                        category_data = compute_averages(raw_data) # Returns list structure
                        print(f"Computed averages for {title}.")

                    if not category_data:
                         print(f"No data obtained for category '{title}' after processing. Skipping plot.")
                         ax.text(0.5, 0.5, 'Processing Failed', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red')
                         continue

                    # Plot the data for the category
                    lbls, lns = plot_averaged_data_with_range(
                        category_data,
                        x_param,
                        y_param,
                        exp_path=rep_exp_path, # Use representative path
                        ax=ax,
                        y_lim=y_lim,
                        min_max=use_min_max_range,
                        show_labels=(i == 0), # Show Y label only on the first plot
                        show_legend_and_title=False, # Legend/Title handled globally
                        save_plot=False,
                    )

                    # Store legend info from the first successful plot
                    if lns and legend_lines is None:
                        legend_labels, legend_lines = lbls, lns

                    # Show X label only on the middle plot (or adjust as preferred)
                    if i == 1:
                         ax.set_xlabel(x_param.replace('_', ' '))
                    else:
                        ax.set_xlabel('')


                except Exception as e:
                    print(f"Error plotting category '{title}': {e}")
                    import traceback
                    traceback.print_exc()
                    ax.text(0.5, 0.5, 'Plotting Error', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red')

            # --- Final Figure Formatting ---
            x_param_text = x_param.replace('_', ' ')
            y_param_text = y_param.replace('_', ' ')

            sim, _, network, _ = exp_config_from_path(rep_exp_path, convert_network=True)
            subtitle = f"Sim: {sim}, Network: {network}"
            fig.suptitle(
                f'{y_param_text} vs {x_param_text}',
                fontsize=18,
                x=0.05, # Align with left subplot edge
                y=0.97,
                horizontalalignment='left',
                verticalalignment='top')
            fig.text(0.05, 0.92, # Position subtitle below main title
                     subtitle,
                     fontsize=12,
                     horizontalalignment='left',
                     verticalalignment='top')

            # Add the common legend if available
            if legend_lines:
                # Place legend outside the subplots to the right
                fig.legend(legend_lines, legend_labels, ncol=4, loc='upper right', borderaxespad=0.1) # Adjust ncol, loc as needed
            else:
                 print("Warning: No legend information was generated.")


            # Save the combined figure
            if output_path:
                os.makedirs(output_path, exist_ok=True)
                # Simplified filename for the 3-plot comparison
                filename = os.path.join(output_path, f'{ABBREVIATIONS.get(x_param_text, x_param_text)}_{ABBREVIATIONS.get(y_param_text, y_param_text)}_averaged.svg')
                fig.savefig(filename)
                print(f"Saved comparison plot to {filename}")

            # plt.show() # Optionally show plot interactively
            plt.close(fig) # Close the figure to free memory
            plt.clf() # Clear any residual plot state


# --- parameters_to_hsluv, parameters_to_color, get_exp_label remain the same ---
# (Color functions are used by plot_averaged_data_with_range)
# (get_exp_label is used in main to create initial dict)

def parameters_to_hsluv(
        values: list[float],
        value_ranges: list[tuple[float, float]],
        hsluv_ranges: list[tuple[float, float]] = ((0, 360), (0, 100), (0, 100)) # Default HSLuv ranges
        ) -> tuple[float, float, float]:
    """Map values to HSLuv components based on specified ranges."""
    def _map_value(value, val_min, val_max, target_min, target_max):
        # Clip value to its range to avoid extrapolation issues
        value = max(val_min, min(value, val_max))
        # Avoid division by zero if range is zero
        if val_max == val_min:
            return target_min # Or target_max, or midpoint
        normalized = (value - val_min) / (val_max - val_min)
        return normalized * (target_max - target_min) + target_min

    hsluv_components = [
        _map_value(val, *val_range, *hsluv_range)
        for val, val_range, hsluv_range in zip(values, value_ranges, hsluv_ranges)
    ]
    return tuple(hsluv_components) # Return HSLuv tuple

def parameters_to_color(
        values: list[float],
        value_ranges: list[tuple[float, float]],
        hsluv_target_ranges: list[tuple[float, float]] = ((0., 240.), (0., 100.), (35., 85.)) # Example target ranges
        ) -> tuple[float, float, float]:
    """Map three values to an RGB color via HSLuv space based on given ranges."""
    # Ensure inputs have length 3
    if len(values) != 3 or len(value_ranges) != 3 or len(hsluv_target_ranges) != 3:
        raise ValueError("Inputs 'values', 'value_ranges', and 'hsluv_target_ranges' must have length 3.")

    hsluv_color = parameters_to_hsluv(values, value_ranges, hsluv_target_ranges)
    rgb_color = hsluv.hsluv_to_rgb(hsluv_color)
    # Clip RGB values to [0, 1] as hsluv conversion can sometimes slightly exceed
    rgb_color_clipped = tuple(max(0.0, min(1.0, c)) for c in rgb_color)
    return rgb_color_clipped


def get_exp_label(filepath: str) -> str:
    """
    From a given filepath, extract the training mode (disturbed/undisturbed/maxpressure) and the seed.
    """
    # Simplified version, adjust patterns if needed
    basename = os.path.basename(os.path.dirname(os.path.dirname(filepath))) # Gets the exp folder name usually
    mode = "Misc"
    if "undisturbed" in basename:
        mode = "Undisturbed"
    elif "disturbed" in basename:
        mode = "Disturbed"
    elif "maxpressure" in basename:
        mode = "MaxPressure"

    seed_part = basename.split("seed")[-1]
    seed_str = ''.join(filter(str.isdigit, seed_part.split('_')[0].split('/')[0]))

    if seed_str:
        return f"{mode}, seed={seed_str}"
    elif mode == "MaxPressure":
         return "MaxPressure" # No seed for MaxPressure usually
    else:
        return f"{mode}, unknown_seed ({basename})" # Fallback


def main():
    # Prompt user to select a file
    # root = tk.Tk()
    # root.withdraw()
    # filepath = filedialog.askopenfilename(initialdir=r".\data\output_data\tsc\sumo_presslight\sumo1x3")
    # print(filepath)
    
    # filepaths = {
    # # # undisturbed
    #     "Undisturbed, seed=100": "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_undisturbed_seed100_eps30_nn32/logger/2024_04_23-15_17_58_BRF.log", # ** 2.
    #     "Undisturbed, seed=200": "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_undisturbed_seed200_eps30_nn32/logger/2024_04_23-14_38_23_BRF.log", # *** 1.
    #     "Undisturbed, seed=300": "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_undisturbed_seed300_eps30_nn32/logger/2024_04_23-12_31_46_BRF.log", # * 3.
    # # # disturbed
    #     "Disturbed, seed=100": "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_disturbed_seed100_eps30_nn32/logger/2024_04_23-16_49_18_BRF.log", # * 3.
    #     "Disturbed, seed=200": "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_disturbed_seed200_eps30_nn32/logger/2024_04_23-16_14_35_BRF.log", # *** 1.
    #     "Disturbed, seed=300": "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_disturbed_seed300_eps30_nn32/logger/2024_04_23-14_01_29_BRF.log", # ** 2.
    # # maxpressure
    #     "MaxPressure": "data/output_data/tsc/sumo_maxpressure/sumo1x3/exp6_1_maxpressure/logger/2024_04_27-12_55_31_BRF.log",
    # }
    # from agent_comparison_plots import choose_experiments
    # filepaths = choose_experiments()
    # filepaths["MaxPressure"] = "data/output_data/tsc/sumo_maxpressure/sumo1x3/exp6_1_maxpressure/logger/2024_04_27-12_55_31_BRF.log"
    from agent_comparison_plots import choose_experiments
    basepath = os.path.join("data", "output_data", "tsc") # if necessary, add `os.path.dirname("."), ` to the front of paths
    list_filepaths = choose_experiments()
    filepaths: dict[str, str] = {get_exp_label(filepath): filepath for filepath in list_filepaths}
    filepaths["MaxPressure"] = os.path.join(basepath, "sumo_maxpressure/sumo1x3/exp6_1_maxpressure/logger/2024_04_27-12_55_31_BRF.log")
    
    
    plot_comparison_figures(
        files=filepaths,
        x_params=["failure chance", "true positive rate", "false positive rate"],
        y_params=["throughput", "travel_time"],
        y_lims=[(1400, 2900), (60, 300)],
        # y_lims=[(1800, 2900), (60, 280)],
        output_path=os.path.join("data", "output_data", "tsc", "stats2")
    )

if __name__ == '__main__':
    main()
