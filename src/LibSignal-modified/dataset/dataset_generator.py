"""
Given a a flow file (JSON format) for vehicle departures and routes, this module provides tools to generate variations on that.

1. Generate a new flow file with the same vehicles departing in the same order but with uniform spacing between departures
2. Generate a new flow file with random vehicles departing uniformly over time with the same route distribution as the original
3. Generate a new flow file with random vehicles departing at random times with the same route distribution as the original
    At each timestep, sample the number of arrvals from a given function f(t)
"""
import json
import os
import random
import xml.etree.ElementTree as ET

from collections import Counter
from typing import Callable

import numpy as np

# 1. Generate a new flow file with the same vehicles departing in the same order but with uniform spacing between departures
def spread_flow_uniformly_json(flow_data, output_filepath):
    """
    Adjusts vehicle departure times in the flow data to ensure a uniform departure rate across the simulation period,
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
    
    # Duration of the simulation in seconds and interval between vehicle departures for a uniform rate
    simulation_duration = simulation_end - simulation_start
    uniform_interval = simulation_duration / total_vehicles
    
    # Generate new start times based on the uniform interval
    new_start_times = [simulation_start + i * uniform_interval for i in range(total_vehicles)]
    
    # Adjust vehicle start times for a uniform departure rate
    uniform_departure_vehicles = [vehicle.copy() for vehicle in flow_data]
    for i, vehicle in enumerate(uniform_departure_vehicles):
        vehicle['startTime'] = new_start_times[i]
    
    # Save the modified data to the specified output file
    with open(output_filepath, 'w') as file:
        json.dump(uniform_departure_vehicles, file, indent=4)


# 2. Generate a new flow file with random vehicles departing uniformly over time with the same route distribution as the original
def random_time_uniform_flow_json(flow_data: list, output_filepath: str, total_vehicles: int = None):
    """
    Generates a new set of vehicles with uniform departure rates, where paths are randomly selected based on their
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


def create_variations_json(flow_file_path: str):
    """
    Generate variations of the original flow file with uniform vehicle departure rates.

    Args:
        flow_file_path (str): The path to the original flow file in JSON format.
    """
    with open(flow_file_path, 'r') as file:
        flow_data = json.load(file)
    # get filename from path
    flow_file_base_path = flow_file_path[:-len(os.path.basename(flow_file_path))]
    # Generate 1st variation
    uniform_path = os.path.join(flow_file_base_path, 'autogen_uniform_flow.json')
    spread_flow_uniformly_json(
        flow_data,
        uniform_path,
    )
    print(f"Saved uniform flow data to {uniform_path}")
    # Generate 2nd variation
    random_uniform_path = os.path.join(flow_file_base_path, 'autogen_random_uniform_flow.json')
    random_time_uniform_flow_json(
        flow_data,
        random_uniform_path,
    )
    print(f"Saved random uniform flow data to {random_uniform_path}")


def create_variation_xml(
            routes_file_path: str,
            output_file_path: str,
            departure_max_time: int = 3600,
            departure_interval: int = 1,
            schedule_type: str = "uniform",
            vehicle_distribution: str = "same"):

    vehicle_types, vehicles = load_routes_data_xml(routes_file_path)
    # Generate a schedule function based on the schedule type
    mean_departure: float = get_mean_departure_rate_xml(vehicles)
    if schedule_type in ("uniform", "const"):
        schedule: Callable[[float, float], int] = const_schedule_fab(mean_departure)
    elif schedule_type == "sine":
        schedule: Callable[[float, float], int] = sine_schedule_fab(
                period=3600,
                mean=mean_departure,
                amplitude=mean_departure/2,
                
                )
    else:
        raise ValueError("Invalid schedule type. expected one of 'uniform', 'const', 'sine'")
    # Generate a vehicle distribution function based on the vehicle distribution type
    if vehicle_distribution == "same":
        vehicle_distribution_fab: Callable[[str, str], dict[str, str]] = same_distribution_vehicle_fab_xml(vehicles)
    else:
        raise ValueError("Invalid vehicle distribution. expected 'same'")
    # Generate new vehicles using the schedule and vehicle distribution functions
    new_vehicles: list[dict[str, str]] = []
    for t in range(0, departure_max_time, departure_interval):
        new_vehicle_count: int = schedule(t, departure_interval)
        for _ in range(new_vehicle_count):
            new_vehicle: dict[str, str] = vehicle_distribution_fab(t, vehicle_types[0]['id'])
            new_vehicles.append(new_vehicle)
    # Save the new data to the specified output file
    save_xml_data(new_vehicles, vehicle_types, output_file_path)


def get_mean_departure_rate_xml(vehicles: list[dict[str, str]]) -> float:
    """
    Calculate the mean departure rate of vehicles from a given list of vehicle data.

    Args:
        vehicles (list[dict[str, str]]): A list of dictionaries containing vehicle data.

    Returns:
        float: The mean departure rate of vehicles in vehicles per hour.
    """
    depart_times: list[float] = [float(vehicle['depart']) for vehicle in vehicles]
    total_vehicles: int = len(depart_times)
    simulation_start: float = min(depart_times)
    simulation_end: float = max(depart_times)
    simulation_duration: float = simulation_end - simulation_start
    mean_departure_rate: float = total_vehicles / (simulation_duration / 3600)
    return mean_departure_rate

def load_routes_data_xml(routes_file_path: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """
    Load vehicle data from an XML file.

    Args:
        routes_file_path (str): The path to the XML file containing vehicle data.

    Returns:
        list[dict[str, str]]: A list of dictionaries containing vehicle types with their defining properties.
        list[dict[str, str]]: A list of dictionaries containing vehicle data.
    """
    tree: ET.ElementTree = ET.parse(routes_file_path)
    root: ET.Element = tree.getroot()
    vehicle_types: list[dict[str, str]] = []
    vehicles: list[dict[str, str]] = []
    for vtype in root.findall('vType'):
        vehicle_types.append({
            'id': vtype.get('id'),
            'vClass': vtype.get('vClass'),
            'speedDev': vtype.get('speedDev'),
            'length': vtype.get('length'),
            'minGap': vtype.get('minGap')
        })
    
    for vehicle in root.findall('vehicle'):
        veh_data = {
            'id': vehicle.get('id'),
            'type': vehicle.get('type'),
            'depart': vehicle.get('depart'),
            'arrival': vehicle.get('arrival'),
            'route': vehicle.find('route').get('edges').split()
        }
        vehicles.append(veh_data)
    return vehicle_types, vehicles

def save_xml_data(vehicles: list[dict[str, str]], vehicle_types: list[dict[str, str]], file_path: str):
    """Save vehicle data back to XML format with indentation."""
    root = ET.Element('routes')
    for vehicle_type in vehicle_types:
        et_vtype = ET.SubElement(root, 'vType',
                id=vehicle_type['id'],
                vClass=vehicle_type['vClass'],
                speedDev=vehicle_type['speedDev'],
                length=vehicle_type['length'],
                minGap=vehicle_type['minGap'])
    for vehicle in vehicles:
        et_vehilce = ET.SubElement(root, 'vehicle',
                id=vehicle['id'],
                type=vehicle['type'],
                depart=vehicle['depart'])
        et_route = ET.SubElement(et_vehilce, 'route', edges=vehicle['route'])

    # Create a new ElementTree from the root
    tree = ET.ElementTree(root)

    # Writing to the file with pretty printing
    ET.indent(tree, space="    ", level=0)
    tree.write(file_path, encoding='utf-8', xml_declaration=True)

def same_distribution_vehicle_fab_xml(routes_data: list[dict[str, str]]) -> Callable[[str, str], dict[str, str]]:
    """
    Given loadded vehicle data from an xml file, return a function that generates vehicle data with the same distribution as the original. Every time this function is called, data for one vehicle is generated.
    Currently, this function expects only one vehicle type in the flow data. This could be extended to support multiple vehicle types in the future.

    Args:
        routes_data (list[dict[str, str]]): loaded vehicle data from an xml file.

    Returns:
        Callable[[str, str], dict[str, str]]: A function that generates vehicle data with the same distribution as the original.
            vehicle_fab(depart_time: str, type: str) -> dict[str, str]
    """
    vehicle_routes: dict[str, int] = {}
    for vehicle in routes_data:
        route_str: str = " ".join(vehicle['route'])
        vehicle_routes[route_str] = vehicle_routes.get(route_str, 0) + 1
    total_routes: int = len(vehicle_routes)
    route_probabilities: dict[str, float] = {route: count/total_routes for route, count in vehicle_routes.items()}
    def vehicle_fab(depart_time: str, type: str) -> dict[str, str]:
        """
        Generate vehicle data with the same distribution as the original. Given a departure time and vehicle type, the function returns a dictionary with the vehicle's departure time, type, and route.

        Args:
            depart_time (str): departure time in a valid SUMO format (e.g. "HH:MM:SS" or as a machine-readable timestamp in seconds).
            type (str): vehicle type.

        Returns:
            dict[str, str]: A dictionary containing the vehicle's departure time, type, route and a (most likely) unique id.
                IDs are generated as random integers in range [1, 10^8]. Generating more than 10^8 vehicles will always lead to duplicates. ID 0 is left unused to accommodate one vehicle type.
        """
        selected_route_str = random.choices(list(route_probabilities.keys()), weights=route_probabilities.values(), k=1)[0]
        return {
            "depart": str(depart_time),
            "id": str(random.randint(1, 1e8)),
            "type": type,
            "route": selected_route_str,
        }
    return vehicle_fab

def const_schedule_fab(mean: int) -> Callable[[float, float], int]:
    """
    Generate a constant departure rate function.

    Args:
        mean (int): mean departure rate (vehicles/hour).
    
    Returns:
        Callable: A function f(t, dt=1) that returns the number of new vehicles departing at time t during time step dt.
    """
    def const_departure_schedule(t: float, dt: float = 1) -> int:
        """
        Generate the number of new vehicles departing at time t with time step dt. Use a constant arrival rate.

        Args:
            t (float): time in seconds.
            dt (float, optional): time step in seconds. Defaults to 1.

        Returns:
            int: the number of new vehicles departing at time t - randomly sampled from a poisson distribution with expected value equal to the expected number of vehicles departing during the time interval dt.
        """
        expected_new_vehicles: float = mean * dt / 3600
        new_vehicle_count: int = int(np.random.poisson(expected_new_vehicles))
        return new_vehicle_count
    return const_departure_schedule

def sine_schedule_fab(
        period: float = 3600,
        amplitude: float = 200,
        mean: float = 200,
        phase_seconds: float = 0,
        phase_radians: float = 0,
        clamp=True,
        ) -> Callable[[float, float], int]:
    """
    Generate a sinusoidal departure rate function. For a given time t and time dt since the last call, the function returns the number of new vehicles .

    Args:
        period (float, optional): period of the sine wave. Defaults to 3600 (seconds).
        amplitude (float, optional): maximum deviation from mean (vehicles/hour). Defaults to 200 (vehicles/hour
        mean (float, optional): mean departure rate (vehicles/hour). Defaults to 200 (vehicles/hour).
        phase_seconds (float, optional): phase shift in seconds. Defaults to 0.
        phase_radians (float, optional): phase shift in radians. Defaults to 0.
        clamp (bool, optional): If True, the departure rate is clamped to be non-negative. Defaults to True.
    
    Returns:
        Callable: A function f(t, dt=1) that returns the number of new vehicles departing at time t during time step dt.
        
    """
    def sine_departure_schedule(t: float, dt: float = 1) -> int:
        """
        Generate the number of new vehicles departing at time t with time step dt. Use a sinusoidal arrival rate. (parameters defined in the outer function)

        Args:
            t (float): time in seconds.
            dt (float, optional): time step in seconds. Defaults to 1.

        Returns:
            int: the number of new vehicles departing at time t - randomly sampled from a poisson distribution with expected value equal to the expected number of vehicles departing during the time interval dt.
        """
        departure_rate: float = \
            mean + amplitude * np.sin((2 * np.pi + phase_radians) * (t + phase_seconds) / period)
        if clamp:
            departure_rate = max(0, departure_rate)
        expected_new_vehicles: float = departure_rate * dt / 3600
        new_vehicle_count: int = int(np.random.poisson(expected_new_vehicles))
        return new_vehicle_count
    return sine_departure_schedule


if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    # file picker
    # print("Select the JSON flow file to generate variations from.")
    # flow_file_path = filedialog.askopenfilename(
    #     title="Select JSON flow file",
    #     filetypes=[("JSON files", "*.json")],
    # )
    # create_variations_json(flow_file_path)
    
    print("Select the XML routes file to generate variations from.")
    routes_file_path = filedialog.askopenfilename(
        title="Select XML routes file",
        filetypes=[("XML files", "*.rou.xml")],
    )
    if not routes_file_path:
        print("No file selected. Stopping program.")
        exit()
    routes_base_name = os.path.basename(routes_file_path)
    routes_base_path = routes_file_path[:-len(routes_base_name)]
    routes_base_name = routes_base_name.strip(".rou.xml")
    output_path: str = routes_base_path + routes_base_name + "_synthetic_sine.rou.xml"
    create_variation_xml(
            routes_file_path,
            output_path,
            departure_max_time=3600,
            departure_interval=1,
            schedule_type="sine",
            vehicle_distribution="same",
            )
    print(f"Saved synthetic data to {output_path}")
