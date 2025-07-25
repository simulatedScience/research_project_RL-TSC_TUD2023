"""
This module implements correction of routes in a Sumo .rou.xml file by fixing vehicles that spawn or despawn in the network interior. See `cologne_3_corrections.md` for details.

Authors: Sebastian Jost, GitHub CoPilot & OpenAI-o4-mini based on human-written instructions in `cologne_3_corrections.md`
"""

# Standard library imports
import random
import xml.etree.ElementTree as ET
from tkinter import filedialog
import tkinter as tk

def load_edge_lists() -> tuple[set[str], set[str]]:
    """
    Returns the sets of incoming and outgoing lane IDs as defined for the Cologne3 network.

    Returns:
        A tuple of (incoming_edges, outgoing_edges).
    """
    incoming = {
        "31864804", "31864804", "241660957#0", "241660957#0",
        "-31800061#0", "-41910185#2", "4045330", "-130160207#0",
        "-241660955#17", "-241660955#17", "41910184", "4045329#5",
        "4999334", "41910186", "319261593#12", "319261593#12",
        "-5229966#3"
    }
    outgoing = {
        "-31864804", "-31864804", "4145590#0", "4145590#0",
        "31800061#0", "4045332#0", "41910185#0", "-4045330",
        "4145589#0", "130160207#0", "241660955#17", "241660955#17",
        "-41910184", "-4045329#5", "-4999334", "-41910186",
        "4999331#0", "4999331#0"
    }
    return incoming, outgoing

def split_vehicles_by_spawn_and_despawn(
    tree: ET.ElementTree,
    incoming: set[str],
    outgoing: set[str]
) -> tuple[list[ET.Element], list[ET.Element], list[ET.Element], list[ET.Element]]:
    """
    Split vehicle elements by spawn and despawn locations.

    Args:
        tree (ET.ElementTree): Parsed XML tree of routes.
        incoming (set[str]): set of incoming lane IDs.
        outgoing (set[str]): set of outgoing lane IDs.

    Returns:
        invalid_spawn, invalid_despawn, valid_spawn, valid_despawn lists of vehicle elements.
    """
    invalid_spawn: list[ET.Element] = []
    invalid_despawn: list[ET.Element] = []
    valid_spawn: list[ET.Element] = []
    valid_despawn: list[ET.Element] = []

    for veh in tree.findall('vehicle'):
        route = veh.find('route')
        edges = route.get('edges').split()
        spawn_edge = edges[0]
        despawn_edge = edges[-1]

        if spawn_edge not in incoming:
            invalid_spawn.append(veh)
        else:
            valid_spawn.append(veh)

        if despawn_edge not in outgoing:
            invalid_despawn.append(veh)
        else:
            valid_despawn.append(veh)

    print(f"Found {len(invalid_spawn)} invalid spawn vehicles,\n      {len(invalid_despawn)} invalid despawn vehicles.")
    print(f"Found {len(valid_spawn)} valid spawn vehicles,\n      {len(valid_despawn)} valid despawn vehicles.")

    return invalid_spawn, invalid_despawn, valid_spawn, valid_despawn


def extract_extensions_despawn(
        source_vehicles: list[ET.Element],
    ) -> dict[str, list[list[str]]]:
    """
    Extract route extensions from valid despawn vehicles:
    For each interior edge, collect valid routes from that edge to an outgoing one as a list of edges.

    Args:
        source_vehicles (list[ET.Element]): Vehicles that are valid for despawn (outgoing).

    Returns:
        (dict[str, list[list[str]]]): Mapping from edge ID to a list of valid extensions (list of edge sequences).
    """
    extensions_map: dict[str, list[list[str]]] = {}
    for veh in source_vehicles:
        # Extract the route edges from the vehicle element
        route = veh.find('route').get('edges').split()
        for i, key_edge in enumerate(route[:-1]):
            # Get the remaining edges after the key edge
            remaining_route = route[i + 1:]
            # Store the extension in the dictionary:
            # key_edge is an interior edge,
            # values are a list of other edges that lead to an outgoing edge.
            if not key_edge in extensions_map:
                extensions_map[key_edge] = []
            extensions_map[key_edge].append(remaining_route)
    print(f"\nExtracted extension for {len(extensions_map)} interior despawn edges.")
    for edge, extensions in extensions_map.items():
        n_unique = len(set(tuple(ext) for ext in extensions))
        print(f"Edge {edge:>13} has {len(extensions):>3} extensions with {n_unique:>2} unique combinations.")
    return extensions_map

def extract_extensions_spawn(
        source_vehicles: list[ET.Element],
    ) -> dict[str, list[list[str]]]:
    """
    Extract route extensions from valid spawn vehicles:
    For each interior edge, collect valid routes from an incoming one to that edge as a list of edges.
    
    Args:
        source_vehicles (list[ET.Element]): Vehicles that are valid for spawn (incoming).
    
    Returns:
        (dict[str, list[list[str]]]): Mapping from edge ID to a list of valid extensions (list of edge sequences).
    """
    extensions_map: dict[str, list[list[str]]] = {}
    for veh in source_vehicles:
        # Extract the route edges from the vehicle element
        route = veh.find('route').get('edges').split()
        for i, key_edge in enumerate(route[1:], start=1):
            # Get the remaining edges before the key edge
            remaining_route = route[:i]
            # Store the extension in the dictionary:
            # key_edge is an interior edge,
            # values are a list of other edges that lead to an incoming edge.
            if not key_edge in extensions_map:
                extensions_map[key_edge] = []
            extensions_map[key_edge].append(remaining_route)
    print(f"\nExtracted extension for {len(extensions_map)} interior spawn edges.")
    for edge, extensions in extensions_map.items():
        n_unique = len(set(tuple(ext) for ext in extensions))
        print(f"Edge {edge:>13} has {len(extensions):>3} extensions with {n_unique:>2} unique combinations.")
    return extensions_map

def fix_routes(
    invalid_spawn: list[ET.Element],
    invalid_despawn: list[ET.Element],
    spawn_exts: dict[str, list[list[str]]],
    despawn_exts: dict[str, list[list[str]]]
) -> None:
    """
    Modify the XML vehicle routes in-place to fix invalid spawns and despawns.

    Args:
        invalid_spawn (list[ET.Element]): Vehicles to fix by appending a spawn extension.
        invalid_despawn (list[ET.Element]): Vehicles to fix by prepending a despawn extension.
        spawn_exts (dict[str, list[list[str]]]): Mapping from spawn edges to spawn extensions.
        despawn_exts (dict[str, list[list[str]]]): Mapping from despawn edges to despawn extensions.
    """
    # randomly extend the start of the route for invalid spawn vehicles.
    extend_spawn_count: int = 0
    extend_despawn_count: int = 0
    for vehicle in invalid_spawn:
        route_elem = vehicle.find('route')
        edges = route_elem.get('edges').split()
        spawn_edge = edges[0]
        extensions = spawn_exts.get(spawn_edge)
        if extensions:
            added_edges = random.choice(extensions)
            # add new edges to the start of the route
            new_edges = added_edges + edges
            route_elem.set('edges', ' '.join(new_edges))
            extend_spawn_count += 1
            # print(f"Extended spawn route for vehicle {vehicle.get('id')} with {len(added_edges)} edges.")
    # randomly extend the end of the route for invalid despawn vehicles.
    for vehicle in invalid_despawn:
        route_elem = vehicle.find('route')
        edges = route_elem.get('edges').split()
        despawn_edge = edges[-1]
        extensions = despawn_exts.get(despawn_edge)
        if extensions:
            added_edges = random.choice(extensions)
            # add new edges to the end of the route
            new_edges = edges + added_edges
            route_elem.set('edges', ' '.join(new_edges))
            extend_despawn_count += 1
            # print(f"Extended despawn route for vehicle {vehicle.get('id')} with {len(added_edges)} edges.")
    print(f"\nExtended {extend_spawn_count} spawn routes and {extend_despawn_count} despawn routes with randomized extensions.")

def process_routes(
    input_file: str,
    output_file: str
) -> None:
    """
    Process the input .rou.xml file, fixing invalid spawn/despawn vehicles, and write the result.

    Args:
        input_file (str): Path to the original .rou.xml file.
        output_file (str): Path to write the corrected .rou.xml.
    """
    incoming, outgoing = load_edge_lists()
    tree = ET.parse(input_file)
    root = tree.getroot()

    invalid_spawn, invalid_despawn, valid_spawn, valid_despawn = \
        split_vehicles_by_spawn_and_despawn(tree, incoming, outgoing)

    spawn_exts = extract_extensions_spawn(valid_spawn)
    despawn_exts = extract_extensions_despawn(valid_despawn)

    fix_routes(invalid_spawn, invalid_despawn, spawn_exts, despawn_exts)

    tree.write(output_file, encoding='utf-8', xml_declaration=True)


def main() -> None:
    """
    Entry point for the script.
    Uses file dialogs to select input and output files, then runs the route processing.
    """
    # Create a root window and hide it
    root = tk.Tk()
    root.withdraw()
    
    # Select input file
    input_file = filedialog.askopenfilename(
        title="Select input .rou.xml file",
        filetypes=[("XML files", "*.xml"), ("Route files", "*.rou.xml"), ("All files", "*.*")]
    )
    
    if not input_file:
        print("No input file selected. Exiting.")
        return
    
    # Select output file
    output_file = filedialog.asksaveasfilename(
        title="Save corrected .rou.xml file as",
        defaultextension=".xml",
        filetypes=[("XML files", "*.xml"), ("Route files", "*.rou.xml"), ("All files", "*.*")]
    )
    
    if not output_file:
        print("No output file selected. Exiting.")
        return
    
    print(f"Processing routes from: {input_file}")
    print(f"Saving corrected routes to: {output_file}")
    
    process_routes(input_file, output_file)
    print("Route processing completed successfully!")


if __name__ == "__main__":
    main()
