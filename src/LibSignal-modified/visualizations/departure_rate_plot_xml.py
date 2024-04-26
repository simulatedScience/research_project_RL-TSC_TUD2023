import os
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt

def parse_xml(file_path: str, normalize_time: bool = False) -> np.ndarray:
    """
    Parse an XML file to extract vehicle departure times.

    Args:
        file_path (str): Path to the XML file.
        normalize_time (bool): Whether to normalize the departure times by subtracting the first departure time.

    Returns:
        np.ndarray: Array of departure times.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    departure_times = []

    for vehicle in root.findall('vehicle'):
        depart = vehicle.get('depart')
        if depart is not None:
            departure_times.append(float(depart))

    departure_times = np.array(departure_times)
    if normalize_time:
        departure_times -= departure_times[0]

    return departure_times

def calculate_departure_rate(departure_times: np.ndarray, window_length: int, step_size: int = 1) -> tuple:
    """
    Calculate the departure rate of vehicles using a centered sliding window approach.

    Args:
        departure_times (np.ndarray): Array of vehicle departure times.
        window_length (int): Length of the sliding window in seconds.
        step_size (int): Step size for each interval in seconds.

    Returns:
        tuple: Tuple containing the times and departure rates.
    """
    max_time = np.max(departure_times)
    time_bins = np.arange(0, max_time + step_size, step_size)
    counts, _ = np.histogram(departure_times, bins=time_bins)
    
    window_bins = int(window_length / step_size)
    windowed_counts = np.convolve(counts, np.ones(window_bins), 'valid') / window_length * 3600

    # Avoid using start and end points where the window isn't fully filled
    half_window = window_bins // 2
    valid_times = time_bins[half_window:-half_window]
    return valid_times, windowed_counts

def plot_departure_rates(files: list, window_length: int, normalize_time: bool, labels: list = None) -> None:
    """
    Plot the departure rates for multiple XML files.

    Args:
        files (list): List of file paths to the XML files.
        window_length (int): Length of the sliding window in seconds.
        normalize_time (bool): Whether to normalize the departure times.
        labels (list): Optional list of labels for the plots.
    """
    plt.figure(figsize=(9, 6))
    for idx, file_path in enumerate(files):
        label = labels[idx] if labels is not None and idx < len(labels) else os.path.basename(file_path)
        departure_times = parse_xml(file_path, normalize_time=normalize_time)
        if len(departure_times) == 0:
            continue
        times, rates = calculate_departure_rate(departure_times, window_length)
        plt.plot(times, rates, label=label)

    plt.title('Departure Rate of Vehicles')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Departure Rate (vehicles/hour)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    tk.Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filepaths = filedialog.askopenfilenames(title="Select .rou.xml flow file", filetypes=[("XML files", "*.rou.xml")])
    print("Selected files:")
    for path in filepaths:
        print(" "*4 + path, end=",\n")
    labels = ["synthetic" if "synth" in os.path.basename(path) else "real-world" for path in filepaths]
    window_length = 300  # Sliding window length in seconds (10 minutes)
    normalize_time = True  # Whether to normalize by subtracting the start time of the first vehicle

    plot_departure_rates(
        filepaths,
        labels=labels,
        window_length=window_length,
        normalize_time=normalize_time)

    # Parameters
    # filepaths = [
    #     "/path/to/your/cologne3.rou.xml",
    #     "/path/to/your/cologne3_synthetic_const.rou.xml",
    #     "/path/to/your/cologne3_synthetic_sine.rou.xml"
    # ]
