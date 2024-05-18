# standard library imports
import time
# third-party imports
# import libsumo
# import traci_/libsumo_
import traci as libsumo

# Define your SUMO configuration file (.sumocfg)
sumo_config: str = "sumo_config.sumocfg"
# Define the scale factor for the traffic
traffic_scale_factor: float = 3.0

# Define a function to control traffic signals based on demand
def demand_based_signal_control(min_green_time=1000, min_red_time=500, yellow_time=300, switch_at_n_veh=500):
    # Get all traffic light ids
    traffic_light_ids = libsumo.trafficlight.getIDList()

    # Predefined phases
    phases = ['GGggrrrrGGggrrrr',
              'yyyyrrrryyyyrrrr',
              'rrrrGGggrrrrGGgg',
              'rrrryyyyrrrryyyy']

    for tl_id in traffic_light_ids:
        # Get all controlled lanes of the traffic light
        controlled_lanes = libsumo.trafficlight.getControlledLanes(tl_id)
        # Calculate the number of halted vehicles at the traffic light
        halted_vehicle_count = sum([libsumo.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes])
        # Get the current phase index
        current_phase_index = libsumo.trafficlight.getPhase(tl_id) % len(phases)
        # Get the time spent in the current phase
        time_in_current_phase = libsumo.trafficlight.getPhaseDuration(tl_id)
        if time_in_current_phase < yellow_time:
            continue
        # Determine the next phase of the traffic light based on the number of halted vehicles
        if 'y' in phases[current_phase_index]:
            # If the current phase is yellow, wait for the yellow time before switching to the next phase
            if time_in_current_phase >= yellow_time:
                next_phase_index = (current_phase_index + 1) % len(phases)
                libsumo.trafficlight.setPhase(tl_id, next_phase_index)
                libsumo.trafficlight.setPhaseDuration(tl_id, 0)  # Reset the phase duration
        elif halted_vehicle_count > switch_at_n_veh and time_in_current_phase >= min_green_time:  
            # If there are more than `switch_at_n_veh` vehicles waiting and the minimum green time has passed, switch to the next phase
            next_phase_index = (current_phase_index + 1) % len(phases)
            print(f"Switching to the next phase at traffic light {tl_id} with {halted_vehicle_count} vehicles 1")
            libsumo.trafficlight.setPhase(tl_id, next_phase_index)
            libsumo.trafficlight.setPhaseDuration(tl_id, 0)  # Reset the phase duration
        elif halted_vehicle_count <= switch_at_n_veh and time_in_current_phase >= min_red_time:
            # If there are `switch_at_n_veh` or fewer vehicles waiting and the minimum red time has passed, switch to the next phase
            print(f"Switching to the next phase at traffic light {tl_id} with {halted_vehicle_count} vehicles 2")
            next_phase_index = (current_phase_index + 1) % len(phases)
            libsumo.trafficlight.setPhase(tl_id, next_phase_index)
            libsumo.trafficlight.setPhaseDuration(tl_id, 0)  # Reset the phase duration



# Run the simulation using the configuration file
# libsumo.start(["sumo", "-c", sumo_config])
libsumo.start(["sumo-gui", "-c", sumo_config])
print(f"started simulation with config file: {sumo_config}")

libsumo.simulation.setScale(traffic_scale_factor)

# Initialize a dictionary to store waiting times
route_waiting_times = {}
route_vehicle_counts = {}

start_time: float = time.perf_counter()
# Run the simulation step by step
while libsumo.simulation.getMinExpectedNumber() > 0:
    libsumo.simulationStep()  # Advance one step in the simulation
    demand_based_signal_control()  # Control the traffic signal
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