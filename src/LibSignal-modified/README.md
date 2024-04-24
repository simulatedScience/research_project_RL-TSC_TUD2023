# Introduction
This section of the repository provides the [LibSignal](https://github.com/DaRL-LibSignal/LibSignal/tree/master) library with some modifications and additions. These focus on investigating how various agents react to sensor failures or noisy sensor outputs - a very common occurance in the real world.

This repo provides OpenAI Gym(nasium) compatible environments for traffic light control scenario and several baseline methods. 

Environments include single intersetion (single-agent) and multi intersections (multi-agents) with different road network and traffic flow settings.

Baselines include traditional Taffic Signal Control algorithms and reinforcement learning based methods.

LibSignal is a cross-simulator environment that provides multiple traditional and Reinforcement Learning models in traffic control tasks. Currently, SUMO, CityFlow, and CBEine simulation environments are supported. Conversion between SUMO and CityFlow is carefully calibrated.

# Changes to the LibSignal library

### Noise model
The main addition compared to the original LibSignal library is the addition of sensor noise and sensor failure simulation. These are implemented in `generator/lane_vehicles.py` using the new functions `simulate_true_positives`, `simulate_false_positives` and `deterministic_random` to modify the existing `generate` method of `LaneVehicleGenerator`.

### Accommodating noise model
Adding this noise model required changing many other files called between `run.py` and `lane_vehicle.py` to pass on and store the noise parameters `fpr`, `tpr`, `failure_chance` and `random_seed`.

### Dataset variations
The original library trained and tested agents on the same datasets. While my experiments have shown this does not lead to overfitting on the training data, I added the possibility to generate new, synthetic datasets to train on those instead. (See `dataset/dataset_generator.py` -> `create_variation_xml`)

### Testing
To evaluate agents on different noise settings, I also modified `run_tests.py` to automatically test given agents on a grid of noise settings, repeat the tests multiple times and store the results.

These functions generate new datasets with sinusoidal or constant arrival rates with the same distribution of vehicles routes as a given original dataset.


### Visualizations
I also added a new folder `visualizations` that contains several scripts to visualize many aspects of the experiments. These include visualizations for:
- various metrics during the training process -> `train_data_plotting.py`
- test performance according to various metrics -> `test_data_plotting.py`
- comparison of several agents -> `agent_comparison_plots.py`
- vehicle departure rate of a given dataset -> `arrival_rate plot.py` (only for JSON datasets, but we use .rou.xml)

# Start

## Run Model Pipeline
This modification of the [LibSignal](https://github.com/DaRL-LibSignal/LibSignal/tree/master) Library is tested by executing `run.py` and setting configurations within that file. The settings are defined at the bottom of the file. Starting experiments via a command-line like the original LibSignal library is not guaranteed to work, but can easily made work again by changing the last few lines of `run.py`.

# Maintaining plan

I do not have specific plans to continue development of this LibSignal modifications after November 2023.

# Original LibSignal

Please see the original [LibSignal](https://github.com/DaRL-LibSignal/LibSignal/tree/master) repository, [website](https://darl-libsignal.github.io/) or the assoiated [paper](https://arxiv.org/abs/2211.10649) for further information. The original Documentation can be found [here](https://darl-libsignal.github.io/LibSignalDoc/).