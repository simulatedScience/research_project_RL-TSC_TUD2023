"""
This script runs a simulation using the SUMO simulator and libsumo.
During the simulation it records the waiting times for each route, then averages them and prints the results.

Author: GPT-4 (03.05.2023) & Sebastian Jost
"""
# standard library imports
import time
# third-party imports
import libsumo

# Define your SUMO configuration file (.sumocfg)
sumo_config: str = "sumo_config.sumocfg"
# Define the scale factor for the traffic
traffic_scale_factor: float = 3.0

# Run the simulation using the configuration file
libsumo.start(["sumo", "-c", sumo_config])

libsumo.simulation.setScale(traffic_scale_factor)

# Initialize a dictionary to store waiting times
route_waiting_times = {}
route_vehicle_counts = {}

start_time: float = time.perf_counter()
# Run the simulation step by step
while libsumo.simulation.getMinExpectedNumber() > 0:
    libsumo.simulationStep() # Advance one step in the simulation
    # Record waiting times for each route
    vehicle_ids = libsumo.vehicle.getIDList()
    for vehicle_id in vehicle_ids:
        route_id = libsumo.vehicle.getRouteID(vehicle_id)
        waiting_time = libsumo.vehicle.getWaitingTime(vehicle_id)

        if route_id not in route_waiting_times:
            route_waiting_times[route_id] = waiting_time
            route_vehicle_counts[route_id] = 1
        else:
            route_waiting_times[route_id] += waiting_time
            route_vehicle_counts[route_id] += 1

# Close the libsumo connection
libsumo.close()
end_time: float = time.perf_counter()
print(f"Simulation time: {end_time - start_time:.2f} seconds")

# Calculate average waiting times for each route
route_avg_waiting_times = {}
for route_id, total_waiting_time in route_waiting_times.items():
    route_avg_waiting_times[route_id] = total_waiting_time / route_vehicle_counts[route_id]

print("Average waiting times for each route:")
for route_id, avg_waiting_time in sorted(route_avg_waiting_times.items(), key=lambda x: x[1]):
    print(f"{route_id:<12}: {avg_waiting_time:.2f} seconds")
