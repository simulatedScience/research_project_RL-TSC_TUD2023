# Introduction
This section of the repository provides the [LibSignal](https://github.com/DaRL-LibSignal/LibSignal/tree/master) library with minor modifications and additions. These focus on investigating how various agents react to sensor failures or noisy sensor outputs - a very common occurance in the real world.

This repo provides OpenAI Gym(nasium) compatible environments for traffic light control scenario and several baseline methods. 

Environments include single intersetion (single-agent) and multi intersections (multi-agents) with different road network and traffic flow settings.

Baselines include traditional Taffic Signal Control algorithms and reinforcement learning based methods.

LibSignal is a cross-simulator environment that provides multiple traditional and Reinforcement Learning models in traffic control tasks. Currently, SUMO, CityFlow, and CBEine simulation environments are supported. Conversion between SUMO and CityFlow is carefully calibrated.

# Start

## Run Model Pipeline
This modification of the [LibSignal](https://github.com/DaRL-LibSignal/LibSignal/tree/master) Library is tested by executing `run.py` and setting configurations within that file. The settings are defined at the bottom of the file. Starting experiments via a command-line like the original LibSignal library is not guaranteed to work, but can easily made work again by changing the last few lines of `run.py`.

# Maintaining plan

I do not have specific plans to continue development of this LibSignal modifications after November 2023.

# Original LibSignal

Please see the original [LibSignal](https://github.com/DaRL-LibSignal/LibSignal/tree/master) repository, [website](https://darl-libsignal.github.io/) or the assoiated [paper](https://arxiv.org/abs/2211.10649) for further information. The original Documentation can be found [here](https://darl-libsignal.github.io/LibSignalDoc/).