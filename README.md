# Research Project: Reinforcement Learning for Traffic Signal Control (@TUD 2023)

This repository contains work for a reasearch project by Sebastian Jost at TU-Dresden in 2023.

## Repository structure
There are some miscellaneous files in the root directory for now, those will be moved to a more appropriate location when the project progresses.
These currently include overviews of related literature (`papers overview.pptx` & `reading_summaries.md`) and notes on planned workflow (`sumo_rl_tsc_workflow.html`).

The code and road networks can be found in `src/`. It is likely that the networks will eventually be moved to a separate folder, but for now they are in the same folder as the code.
Some introductory examples can be found in `src/sumo_python_intro`: a single intersection setup and one with four intersections arranged in a square.

## Future plans
We plan to implement Reinforcement Learning algorithms for Traffic Signal Control in SUMO using Python. The algorithms will be tested on a variety of road networks.

## Installation

1. Clone this repository from [GitHub](https://github.com/simulatedScience/research_project_RL-TSC_TUD2023)
2. Install SUMO (Simulation of Urban MObility) from [sumo.dlr.de](https://sumo.dlr.de/docs/Downloads.html)
3. Install python libraries using `pip install -r src/LibSignal-modified/requirements.txt`  
   (You may need to explicitly specifiy the target path when using a virtual environment, e.g. `pip install -t .venv/Lib/site-packages -r src/LibSignal-modified/requirements.txt`)
4. Install pytorch for your system from [pytorch.org](https://pytorch.org/get-started/locally/)  
   (You may need to install a different version of pytorch depending on your system and CUDA version. TO enable GPU support, you need to have a compatible GPU and [CUDA](https://developer.nvidia.com/cuda-zone) installed. At time of testing, the recommended version is [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64))


## Running the code

The relevant code can be found in `src/LibSignal-modified`.

### Training
1. find the files `configs/tsc/base.yml` and the one corresponding to the agent you want to train (e.g. `configs/tsc/presslight.yml`)
2. in `configs/tsc/[agent].yml`, set the `train_model` parameter to `True`  
   if `base.yml` is imported into the agent's config, the agent overwrites any identically named setting in `base.yml`
3. set the number of episodes to train for to the desired amount in `base.yml` or the agent's config (keep in mind the overwriting mentioned above)
4. find `run.py` in the root directory.
5. set the desired parameters there, most notably the different sensor failure rates, experiment name, road network and agent to use.  
   The agent name must match the name of the agent's config file.
6. run `run.py`
7. find the results in `data/output_data/tsc/sumo_[agent]/[network]/[experiment_name]`. (e.g. `data/output_data/tsc/sumo_presslight/sumo1x3/exp_9_disturbed_100`)
8. Within the experiment folder from step 7, you can find logging data in `logger`

### Testing
1. find the files `configs/tsc/base.yml` and the one corresponding to the agent you want to test (e.g. `configs/tsc/presslight.yml`)
2. ensure that the `train_model` parameter is set to `False` in the agent's config
3. set both `load_model` and `test_model` to `True` in the agent's config
4. continue with steps 4-6 from the training section. The program will automatically find the latest model in the given experiment to test with.  
   Note, that you can test with different sensor failure rates than you trained with. Passing lists of sensor failure rates will test all possible combinations of them (kartesian product).  

### Parameters used in paper
For training the model on noisy data, we used the following parameters:
- sensor failure rate (`failre_chance`): 0.1
- true positive rate (`tpr`): 0.8
- false positive rate (`fpr`): 0.15
