
# TODOs for publication

- Visualize learned policy of the RL agents
    1. show policy in SUMO simulation for manual inspection & sanity check
    2. produce graphs where policies can be compared -> gain insights how the policy changes when training on noisy data

- check any new research on the topic

- **✔ Checked:** double-check which learning algorithm is used in the LibSignal code
  -> The Implementation of PressLight uses Double DQN (DDQN).  
  Actor is update using Bellman equation every `update_model_rate` steps.  
  Target network is updated every `update_target_rate` steps by copying the actor's weights.  
  Is DDQN still a good enough choice of algorithm in 2024? It's very simple and easy to implement.

- Verify that training and test dataset are different. Repeat experiments if not.
Options to get more data:
    - Train on synthetic data with different arrival rates, test on real data
    - reverse time and/or destinations of first dataset for more diverse data
  
  **Notes:**  
  Vehicles are added into the simulation in `world/world_sumo_disturbed.py` `ln. 532` based on a sumo config.
  Upon initialisation, the class `World` generates a terminal command used to start SUMO. This contains info like the road network file and the route/ flow file. (See `ln. 370` in `world/world_sumo_disturbed.py`) Which files are loaded is determined by the sumo config file found in `configs/sim` (e.g. `sumo1x3.cfg`). This config is loaded (JSON format) to get the relevant file paths.  
  During training, the environment is reset for each episode in `tsc_trainer.train()` (see `self.env.reset()`). This does not indicate any switch of datasets between training and testing. Neither does the similar reset in `tsc_trainer.train_test()`.


- Write instructions for how to reproduce the experiments, add `requirements.txt` etc.

- **✔ CORRECTED:** Verify that the TPR and FPR measured in the simulation are the same as the values that are set.
    - Measure TPR & FPR in simulation to verify that a given setting leads to the expected results
    - Add correction factor for TPR/ FPR. Currently:
    measured incorrect detections=(detected vehicles)/TPR⋅FPR=(incorrect detections)/TPR  
    - The correctness of the simulation can be tested using `src/LibSignal_modified/tpr_test.py`
  In original version, the measured FPR was about half of the expected FPR and depended on the TPR.

- Verify results in a second environment (road network)

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
