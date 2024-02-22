"""
Given a a flow file (JSON format) for vehicle arrivals and routes, this module provides tools to generate variations on that.

1. Generate a new flow file with the same vehicles arriving in the same order but with uniform spacing between arrivals
2. Generate a new flow file with random vehicles arriving uniformly over time with the same route distribution as the original
3. Generate a new flow file with random vehicles arriving at random times with the same route distribution as the original
    At each timestep, sample the number of arrvals from a given function f(t)
"""
import json
import os
import random

from collections import Counter
from typing import Callable

import numpy as np

# 1. Generate a new flow file with the same vehicles arriving in the same order but with uniform spacing between arrivals
def spread_flow_uniformly(flow_data, output_filepath):
    """
    Adjusts vehicle arrival times in the flow data to ensure a uniform arrival rate across the simulation period,
    keeping all other vehicle attributes and routes the same.
    
    Args:
        flow_data (list): The loaded JSON data representing the original flow of vehicles.
        output_filepath (str): The path where the modified JSON data should be saved.
    """
    # Calculate the total simulation duration and the number of vehicles
    total_vehicles = len(flow_data)
    start_times = [vehicle['startTime'] for vehicle in flow_data]
    simulation_start = min(start_times)
    simulation_end = max(start_times)  # Using max of startTime for a uniform distribution
    
    # Duration of the simulation in seconds and interval between vehicle arrivals for a uniform rate
    simulation_duration = simulation_end - simulation_start
    uniform_interval = simulation_duration / total_vehicles
    
    # Generate new start times based on the uniform interval
    new_start_times = [simulation_start + i * uniform_interval for i in range(total_vehicles)]
    
    # Adjust vehicle start times for a uniform arrival rate
    uniform_arrival_vehicles = [vehicle.copy() for vehicle in flow_data]
    for i, vehicle in enumerate(uniform_arrival_vehicles):
        vehicle['startTime'] = new_start_times[i]
    
    # Save the modified data to the specified output file
    with open(output_filepath, 'w') as file:
        json.dump(uniform_arrival_vehicles, file, indent=4)


# 2. Generate a new flow file with random vehicles arriving uniformly over time with the same route distribution as the original
def random_time_uniform_flow(flow_data: list, output_filepath: str, total_vehicles: int = None):
    """
    Generates a new set of vehicles with uniform arrival rates, where paths are randomly selected based on their
    frequency in the original flow data.
    
    Args:
        flow_data (list): The loaded JSON data representing the original flow of vehicles.
        output_filepath (str): The path where the new JSON data should be saved.
        total_vehicles (int, optional): The total number of vehicles to generate. Defaults to None (same as original flow).
    """
    # Extract routes and calculate frequencies
    all_routes = [tuple(vehicle['route']) for vehicle in flow_data]
    route_frequencies = Counter(all_routes)
    total_routes_count = sum(route_frequencies.values())
    route_probabilities = {route: count / total_routes_count for route, count in route_frequencies.items()}

    # Calculate new start times for uniform distribution
    if total_vehicles is None:
        total_vehicles = len(flow_data)
    simulation_start = min(vehicle['startTime'] for vehicle in flow_data)
    simulation_end = max(vehicle['startTime'] for vehicle in flow_data)
    simulation_duration = simulation_end - simulation_start
    uniform_interval = simulation_duration / total_vehicles
    new_start_times = [simulation_start + i * uniform_interval for i in range(total_vehicles)]
    
    # Generate new vehicles with weighted random routes
    weighted_random_vehicles = []
    for start_time in new_start_times:
        selected_route = random.choices(list(route_probabilities.keys()), weights=route_probabilities.values(), k=1)[0]
        new_vehicle = {
            "vehicle": flow_data[0]["vehicle"],  # Reuse the first vehicle's attributes
            "route": list(selected_route),
            "interval": flow_data[0]["interval"],
            "startTime": start_time,
            "endTime": start_time + 2  # Assuming a fixed duration
        }
        weighted_random_vehicles.append(new_vehicle)
    
    # Save the new data to the specified output file
    with open(output_filepath, 'w') as file:
        json.dump(weighted_random_vehicles, file, indent=4)


def create_variations(flow_file_path: str):
    """
    Generate variations of the original flow file with uniform vehicle arrival rates.

    Args:
        flow_file_path (str): The path to the original flow file in JSON format.
    """
    with open(flow_file_path, 'r') as file:
        flow_data = json.load(file)
    # get filename from path
    flow_file_base_path = flow_file_path[:-len(os.path.basename(flow_file_path))]
    # Generate 1st variation
    uniform_path = os.path.join(flow_file_base_path, 'autogen_uniform_flow.json')
    spread_flow_uniformly(
        flow_data,
        uniform_path,
    )
    print(f"Saved uniform flow data to {uniform_path}")
    # Generate 2nd variation
    random_uniform_path = os.path.join(flow_file_base_path, 'autogen_random_uniform_flow.json')
    random_time_uniform_flow(
        flow_data,
        random_uniform_path,
    )
    print(f"Saved random uniform flow data to {random_uniform_path}")


def sin_schedule_fab(
        period: float = 3600,
        amplitude: float = 200,
        mean: float = 200,
        phase: float = 0,
        clamp=True,
        ) -> Callable[[float, float], int]:
    """
    Generate a sinusoidal arrival rate function.

    Args:
        period (float, optional): period of the sine wave. Defaults to 3600 (seconds).
        amplitude (float, optional): maximum deviation from mean (vehicles/hour). Defaults to 200 (vehicles/hour
        mean (float, optional): mean arrival rate (vehicles/hour). Defaults to 200 (vehicles/hour).
        phase (float, optional): phase shift in radians (0 to 2π). Defaults to 0.
        clamp (bool, optional): If True, the arrival rate is clamped to be non-negative. Defaults to True.
    
    Returns:
        Callable: A function f(t, dt=1) that returns the arrival rate at time t and time step dt.
        
    """
    def f(t: float, dt: float = 1) -> int:
        arrival_rate: float = \
            mean + amplitude * np.sin(2 * np.pi * t / period + phase)
        if clamp:
            arrival_rate = max(0, arrival_rate)
        expected_new_vehicles: float = arrival_rate * dt / 3600
        # generate new vehicles with poisson distribution
        new_vehicle_count: int = np.random.poisson(expected_new_vehicles)
        return new_vehicle_count
    return f


if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    # file picker
    print("Select the JSON flow file to generate variations from.")
    flow_file_path = filedialog.askopenfilename(
        title="Select JSON flow file",
        filetypes=[("JSON files", "*.json")],
    )
    create_variations(flow_file_path)