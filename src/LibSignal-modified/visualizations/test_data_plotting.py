import matplotlib.pyplot as plt


def plot_with_third_param_customizable(data, x_param, y_param, third_param, colormap_name='inferno', **kwargs):
    # Extract unique values of the third parameter and sort them
    third_param_values = sorted(list(set([entry[third_param] for entry in data])))
    
    plt.figure(figsize=(10, 6))
    
    # Define a colormap to create a gradient based on third parameter values
    colormap = plt.cm.get_cmap(colormap_name, len(third_param_values))
    
    # Plot data for each unique value of the third parameter
    for idx, third_val in enumerate(third_param_values):
        subset_data = [entry for entry in data if entry[third_param] == third_val]
        sorted_data = sorted(subset_data, key=lambda x: x[x_param])  # Sort data by x_param
        x_values = [entry[x_param] for entry in sorted_data]
        y_values = [entry['metrics'][y_param] for entry in sorted_data]
        plt.plot(x_values, y_values, 'o-', label=f'{third_param}: {third_val}', color=colormap(idx), **kwargs)
    
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(f'{y_param} vs {x_param} (Grouped by {third_param})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Testing the updated function with the new colormap and customizable appearance
plot_with_third_param_customizable(data_list, 'failure_chance', 'throughput', 'noise_chance')
plot_with_third_param_customizable(data_list, 'failure_chance', 'delay', 'noise_chance')
plot_with_third_param_customizable(data_list, 'noise_chance', 'throughput', 'failure_chance')
plot_with_third_param_customizable(data_list, 'noise_chance', 'delay', 'failure_chance')


new_files_2 = [
    "/mnt/data/2023_10_23-13_43_39_BRF_fc0.15_nc1.0_nr0.15.log",
    "/mnt/data/2023_10_23-13_42_13_BRF_fc0.1_nc1.0_nr0.15.log",
    "/mnt/data/2023_10_23-13_40_07_BRF_fc0.05_nc1.0_nr0.15.log",
    "/mnt/data/2023_10_23-13_38_10_BRF_fc0.0_nc1.0_nr0.15.log"
]