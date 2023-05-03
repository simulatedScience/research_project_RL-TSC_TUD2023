# Single-intersection SUMO demonstration
This subfolder contains a few simple examples of using SUMO with Python.
This file documents the process of creating a simulation with SUMO and the Python interface.

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

All steps for building the network, setting up routes and fixed signal timings can also be done through a visual interface: netedit - a program that is shipped with SUMO installations.  
This program has modes for...
- creating nodes and edges,
- creating traffic lights and defining signal timings,
- specifying lane turning restrictions,
- setting traffic routes/ flows,
- and much more.

From there the necessary files can easily be exported and then used in simulations.

## Python interface
Instead of using a terminal or the GUI, the simulation can also be run from a Python script.
For this, the libraries `traci` and `libsumo` are available. Both need to be installed seperately and fulfill the same purpose.
There are a few features that libsumo does not properly support, but it is much faster than traci (on a small test, traci took 60s, while libsumo took 25s). `libsumo` is a drop-in replacement for `traci`, barring a few exceptions.

These libraries can be used to initialize a simulation, run it step by step and retrieve information about the current state of the simulation. With the current state, performance metrics can be measured and used to evaluate the performance of a traffic signal controller.
The Python API enables editing many parameters of the simulations, even while it is running. This can be used to implement dynamic traffic signal control algorithms, dynamic vehicle routes and more.