# Single-intersection SUMO demonstration
This project is a very simple demonstration of the SUMO library using a single intersection with a traffic light.

## Creating a simulation with SUMO
Creating the simulation requires the following steps:
- Create a road network:
  - specify nodes in a `.nod.xml` file
  - specify edges in a `.edg.xml` file
  - limit turning options from each lane in a `.con.xml` file
  - specify traffic signal timings in a `.tll.xml` file
    - for each phase, specify all possible turning options for each line with a letter (r,y,g,R,Y,G)
- compile network files into a `.net.xml` file
  - create a `.netccfg` file to specify the network configuration. Reference the node, edge and signal timing files here.
  - run `netconvert -c [filename].netccfg` (look for "Success" message)
- create a simulation configuration
  - specify routes that vehicles can take in a `.rou.xml` file
  - specify the simulation configuration in a `.sumocfg` file:
    - specify the network file
    - specify the route file
    - specify the simulation duration
- run the simulation either from terminal or visually:
  - `sumo -c [filename].sumocfg`
  - double click the `.sumocfg` file to open the simulation in the SUMO-GUI
  - open the SUMO-GUI and select the `.sumocfg` file, then run the simulation using the play button

All steps for building the network, setting up routes and fixed signal timings can also be done through a visual interface: netedit. A program that is shipped with SUMO installations.
This program has various modes for:
- creating nodes and edges,
- creating traffic lights and specifying signal timings,
- specifying lane turning restrictions
- setting traffic routes/ flows

From there the necessary files can easily be exported and then used in simulations.

## Python interface
Instead of using a terminal or the GUI, the simulation can also be run from a Python script.
For this, the libraries `traci` and `libsumo` are available. Both need to be installed seperately and fulfill the same purpose.
There are a few features that libsumo does not properly support, but it is much faster than traci (on a small test, traci took 60s, while libsumo took 25s). No other code changes were necessary to switch between the two libraries.