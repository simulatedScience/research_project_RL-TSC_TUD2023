"""
Author: Sebastian Jost & GPT-4 (27.10.2023)
"""
import json
import matplotlib.pyplot as plt
from collections import Counter

def calculate_centered_arrival_rate_padded(data, window_length_seconds):
    """
    Calculate the arrival rate of vehicles over time based on a centered window length.
    Pads the result with None for times outside the window range.
    
    Args:
        data (list of dict): List of dictionaries containing vehicle data.
        window_length_seconds (int): Length of the centered window in seconds used to calculate the vehicles per hour.
    
    Returns:
        times (list of int): List of times.
        rates_per_hour (list of float): List of arrival rates (vehicles/hour) corresponding to the times.
    """
    start_times = [entry['startTime'] for entry in data]
    counts = Counter(start_times)
    
    max_time = 3600  # Simulation runs for 3600s
    times = list(range(max_time))
    rates_per_hour = [None] * max_time
    half_window = window_length_seconds // 2
    
    for t in times[half_window:-half_window]:
        current_count = sum(counts.get(time, 0) for time in range(t - half_window, t + half_window))
        rate = (current_count / window_length_seconds) * 3600
        rates_per_hour[t] = rate
        
    return times, rates_per_hour

def plot_centered_arrival_rate(data, window_length_seconds):
    """
    Plot the arrival rate of vehicles over time based on a centered window length.
    
    Args:
        data (list of dict): List of dictionaries containing vehicle data.
        window_length_seconds (int): Length of the centered window in seconds used to calculate the vehicles per hour.
    """
    times, rates_per_hour = calculate_centered_arrival_rate_padded(data, window_length_seconds)
    plt.plot(times, rates_per_hour, label=f'Centered Window Length: {window_length_seconds}s')
    plt.axhline(y=2800, color='r', linestyle='--', label='Total Vehicles: 2800')
    plt.title('Arrival Rate of Vehicles Over Time for Centered Window Length')
    plt.xlabel('Time')
    plt.ylabel('Vehicles per Hour')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    import tkinter as tk
    from tkinter import filedialog
    tk.Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filepath = filedialog.askopenfilename(title="Select JSON flow file", filetypes=[("JSON files", "*.json")])
    with open(filepath, "r") as file:
        data = json.load(file)
    
    # Plot the arrival rate for a centered window length of 300 seconds
    plot_centered_arrival_rate(data, window_length_seconds=300)
