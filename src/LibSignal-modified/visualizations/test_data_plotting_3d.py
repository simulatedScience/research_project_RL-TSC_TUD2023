"""
Visualize the test data using a 3D plot with each failure component as a spactial dimension and the metric represented by the color.

Use a seismic colormap with the FixedTime agent's performance as the white point.
"""

import tkinter as tk
from tkinter import filedialog
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3D plotting
from matplotlib import cm # Colormaps
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from data_reader import read_and_group_test_data, experiment_name, exp_config_from_path, ABBREVIATIONS, get_fixedtime_data


def plot_3d_metric(
        grouped_data,
        metric="travel_time",
        exp_path: str = None,
        fixed_time=None,
        value_range=(60, 280)):
    """
    Plot the given metric in a 3D space with axes as `failure chance`, `true positive rate`, `false positive rate`.
    Uses a seismic colormap with a colorbar. The `fixed_time` indicates the midpoint for the colormap if provided,
    and the value_range sets the bounds for color scaling to blue (min) and red (max).
    
    Args:
        grouped_data (dict): Data grouped by (fc, tpr, fpr) with lists of metrics.
        metric (str): The metric to plot, default is `travel_time`.
        fixed_time (float): The fixed time value to center the colormap, if None it calculates the median.
        value_range (tuple): The minimum and maximum values for the metric to define the color scale.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    # ax.set_facecolor("black")  # Set the background to black
    
    # Prepare data for plotting
    x, y, z, c = [], [], [], []
    for (fc, tpr, fpr), metrics in grouped_data.items():
        if metrics[metric]:  # Ensure there is data for the metric
            avg_metric = sum(metrics[metric]) / len(metrics[metric])  # Average the metric values
            x.append(fc)
            y.append(tpr)
            z.append(fpr)
            c.append(avg_metric)
    
    if not c:  # If there are no data points to plot
        print("No data available for the specified metric.")
        return
    
    # Set the fixed_time if not specified
    if fixed_time is None:
        fixed_time = np.median(c)
    
    # Determine color scale limits based on the specified range
    vmin, vmax = value_range[0], value_range[1]
    
    # Create the scatter plot
    custom_cmap = shiftedColorMap(cm.seismic, midpoint=(fixed_time - vmin) / (vmax - vmin))
    scatter = ax.scatter(x, y, z, c=c, cmap=custom_cmap, vmin=vmin, vmax=vmax, s=150, alpha=1, edgecolors="black")
    
    # set default view
    ax.view_init(elev=12, azim=102)
    
    # Add a color bar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label(metric)
    
    # Set labels
    ax.set_xlabel("Failure Chance")
    ax.set_ylabel("True Positive Rate")
    ax.set_zlabel("False Positive Rate")
    
    # Set the ticks for the x, y, and z axes to the unique values of the plotted points
    ax.set_xticks(np.unique(x))
    ax.set_yticks(np.unique(y))
    ax.set_zticks(np.unique(z))
    
    # Title and show
    plt.title(f"3D Plot of {metric} with color range {value_range} (centered at {fixed_time})")
    # save figure
    sim, method, network, exp_name = exp_config_from_path(exp_path, convert_network=True)
    exp_info = experiment_name(sim, method, network, exp_name)
    
    os.makedirs(os.path.join(exp_path, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(exp_path, 'plots', f'3D_{metric}_{exp_info}.png'))
    # plt.show()
    plt.clf()
    
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
    Function copied from [stackoverflow, @Paul H, 2013](https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib)
    
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap"s dynamic range to be at zero.

    Args
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap"s range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap"s range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {
        "red": [],
        "green": [],
        "blue": [],
        "alpha": []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = LinearSegmentedColormap(name, cdict)
    # # if cmap is not reg
    #     plt.register_cmap(cmap=newcmap)

    return newcmap

def main():
    # # Prompt user to select a file
    # root = tk.Tk()
    # root.withdraw()
    # filepath = filedialog.askopenfilename(initialdir=r".\data\output_data\tsc\sumo_presslight\sumo1x3")
    # print(filepath.split("/LibSignal-modified/")[-1])
    filepaths = [
        # undisturbed
            "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_undisturbed_seed100_eps30_nn32/logger/2024_04_23-15_17_58_BRF.log",
            "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_undisturbed_seed200_eps30_nn32/logger/2024_04_23-14_38_23_BRF.log",
            "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_undisturbed_seed300_eps30_nn32/logger/2024_04_23-12_31_46_BRF.log",
        # disturbed
            "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_disturbed_seed100_eps30_nn32/logger/2024_04_23-16_49_18_BRF.log",
            "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_disturbed_seed200_eps30_nn32/logger/2024_04_23-16_14_35_BRF.log",
            "data/output_data/tsc/sumo_presslight/sumo1x3/exp6_disturbed_seed300_eps30_nn32/logger/2024_04_23-14_01_29_BRF.log",
        # maxpressure
            "data/output_data/tsc/sumo_maxpressure/sumo1x3/exp6_1_maxpressure/logger/2024_04_27-12_55_31_BRF.log",
    ]
    for filepath in filepaths:
        grouped_data = read_and_group_test_data(filepath)
        fixed_time_data = get_fixedtime_data()
        exp_path = filepath.strip(os.path.basename(filepath))[:-len("logger/") ]

        # plot averaged data with ranges
        ylim_travel_time = (60, 280)
        ylim_throughput = (1800, 2900)

        plot_3d_metric(grouped_data, metric="travel_time", exp_path=exp_path, fixed_time=fixed_time_data["travel_time"], value_range=ylim_travel_time)
        plot_3d_metric(grouped_data, metric="throughput", exp_path=exp_path, fixed_time=fixed_time_data["throughput"], value_range=ylim_throughput)

if __name__ == "__main__":
    main()