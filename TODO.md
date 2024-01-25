
# TODOs for publication

- Visualize learned policy of the RL agents
    1. show policy in SUMO simulation for manual inspection & sanity check
    2. produce graphs where policies can be compared -> gain insights how the policy changes when training on noisy data

- check any new research on the topic

- double-check which learning algorithm is used in the LibSignal code

- Verify that training and test dataset are different. Repeat experiments if not.
Options to get more data:
    - Train on synthetic data with different arrival rates, test on real data
    - reverse time and/or destinations of first dataset for more diverse data

- Verify assumptions of statistical models
    - Measure TPR & FPR in simulation to verify that a given setting leads to the expected results
    - Add correction factor for TPR/ FPR. Currently:
    measured incorrect detections=(detected vehicles)/TPR⋅FPR=(incorrect detections)/TPR  

- Verify results in a second environment

- Repeat Experiments many more times to bring down standard deviation in Fig. 10 

- Understand & explain, why the agent trained on disturbed data performs worse on clean data than on disturbed data. (see Fig. 10 travel time chart)

- Test more different agents (FRAP, CoLight)

### On policy visualization

#### Problem:
- Agents trained on noisy data worked much better than expected, we don't know why yet.  
   This could be caused by an undesirable policy that switches between phases constantly or by exploiting certain traffic flow 
    
#### Possible visualizations:
- Video showing agent policy in SUMO simulation
- Graph showing phase of each intersection over time
   - See Fig. 6 in 20201201 Robust RL for TSC:  
      Queue length over time with green direction marked on the x-axis (per intersection)
   - See Fig. 9 & 10 in 20190804 Presslight:  
      Fig. 9: Green wave visualization:  
        Distance travelled by vehicles in one direction over time with intersection phases in direction of travel marked on horizontal stripes
    Fig. 10: Space-time with signal phases:  
    Distance travelled by vehicles in two opposite directions over time with intersection phases in direction of travel marked on horizontal stripes
   - See Fig. 9 in 20180819 Intellilight:  
    Phase time ratio over time for each intersection
Phase over time

## Overview of related work

| Paper Title | Road Network(s) | RL Algorithm | Simulator |
|-------------|-----------------|--------------|-----------|
| [robust RL for TSC](https://doi.org/10.1007/s42421-020-00029-6) | single intersection | double DQN | SUMO |
| [PressLight](https://doi.org/10.1145/3292500.3330949) | Qingdao Rd., Jinan 1x3;  Beaver Ave., State College 1x5;  8-th to 11-th Ave., NYC (4 separate networks), 1x16 each | DQN with  experience replay | cityflow |
| [MPLight /  A thousand Lights](https://doi.org/10.1609/aaai.v34i04.5744) | 4x4 grid;  Manhattan | DQN with (shared)  experience replay | cityflow |
| [FRAP (phase   competition   signal control)](https://doi.org/10.1145/3357384.3357900) | Atlanta 1x5;  Jinan 7 intersections;  Hangzhou 1x6 | DQN  ? likely with   experience replay ? | SUMO |
| [CoLight](https://doi.org/10.1145/3357384.3357902) | synth. Arterial 1x3;  synth. Grid 3x3;  synth. Grid 6x6;  Manhattan 196=7x28;  Gudang Sub-district, Hangzhou, 4x4;  Dongfeng Sub-district, Jinan, 3x4 | unknown -> see code | cityflow |
| [RL Benchmarks for TSC](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/f0935e4cd5920aa6c7c996a5ee53a70f-Abstract-round1.html) | synth. grid 4x4;  synth. Ave. grid 4x4;  Cologne 1x3;  Ingolstadt 1x7 | DQN using Preferred RL library | SUMO |
...
