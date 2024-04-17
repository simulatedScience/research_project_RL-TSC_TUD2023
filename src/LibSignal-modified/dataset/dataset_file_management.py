"""
This module provides functions to load traffic flow data from XML (SUMO) or JSON (cityflow) files.
"""


import xml.etree.ElementTree as ET
from xml.dom import minidom

def load_xml_data(file_path):
    """Load and parse the XML data."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    return root

def extract_vehicle_times_XML(root):
    """Extract vehicle depart times from XML."""
    depart_times = []
    for vehicle in root.findall('.//vehicle'):
        depart_time = float(vehicle.get('depart'))
        depart_times.append(depart_time)
    return depart_times

def parse_vehicle_data(root):
    """Extract vehicle and route data from XML."""
    vehicles = []
    for vtype in root.findall('vType'):
        vehicles.append({
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
    return vehicles

def save_xml_data(vehicles, file_path):
    """Save vehicle data back to XML format with indentation."""
    root = ET.Element('routes')
    for veh in vehicles:
        if 'vClass' in veh:  # It's a vType
            vtype = ET.SubElement(root, 'vType', id=veh['id'], vClass=veh['vClass'], speedDev=veh['speedDev'], length=veh['length'], minGap=veh['minGap'])
        else:  # It's a vehicle
            vehicle = ET.SubElement(root, 'vehicle', id=veh['id'], type=veh['type'], depart=veh['depart'], arrival=veh['arrival'])
            route = ET.SubElement(vehicle, 'route', edges=' '.join(veh['route']))
    
    # Create a new ElementTree from the root
    tree = ET.ElementTree(root)

    # Writing to the file with pretty printing
    ET.indent(tree, space="    ", level=0)
    tree.write(file_path, encoding='utf-8', xml_declaration=True)
    # xml_str = ET.tostring(root, 'utf-8')
    # pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="    ")  # 4 spaces
    # with open(file_path, 'w') as f:
    #     f.write(pretty_xml)