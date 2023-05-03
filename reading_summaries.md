# Summaries of Papers on RL for traffic signal control

## Introduction
Traffic signals, typically found at intersections have some important responsibilities but also introduce challenges. They are meant to keep traffic safe, but should keep the cost of travel low.
Traffic congestion can cause real problems like:
- public service vehicles being slowed down (police, ambulance, fire department)
- delays in transportation of goods
- slow traffic leading to wider than necessary roads being built
- unhappiness of drivers

## paper summaries
### Intellilight
[paper](https://www.kdd.org/kdd2018/accepted-papers/view/intellilight-a-reinforcement-learning-approach-for-intelligent-traffic-ligh)

- proposes two major contributions:
  - **memory palace**: seperate memory into different categories to reduce biases in generated data.
    - group by pairs (phase, action)
    - sample uniformly from categories to learn equally for all situations
  - **phase gate**: depending on the current phase, a different set of dense layers is used in the NN.
    - this makes it easier for the network to change it's behaviour based on the current phase
  - (explicit interpretation of the policy, not just rewards)

- Features:
  - Queue length per lane $L_i \in \mathbb{N}_0$
  - Number of vehicles per lane $V_i \in \mathbb{N}_0$
  - Waiting time of vehicles per lane $W_i \in \mathbb{R}$  
    Waiting time is reset to 0 when the vehicle moves faster than 0.1 m/s.
    $W_i = $ sum of waiting times of all approaching vehicles in lane $i$
  - Image representation of vehicle positions $M\in ??$
  - Current phase $P_c \in \{0,1\}$
  - Next phase $P_n \in \{0,1\}$

- Reward:  
  weighted sum of:
  - Sum of queue lengths: $L_i$
  - Sum of delays $D_i = 1 - \frac{\text{lane speed}}{\text{speed limit}}$
  - Sum of waiting times $W_i$
  - whether phase was changed  
    $C=0$ for keeping phase, $C=1$ for changing phase
  - number $N$ of vehicles that passed intersection since last action
  - total travel time $T$ (in min) of vehicles that passed intersection since last action  
    travel time for each vehicle is the time it spent on approaching lanes
- Training policy: $\varepsilon$-greedy

### Toward A Thouseand Lights: Decentralized Deep RL for Large-Scale TRaffic Signal Control
[paper](https://chacha-chen.github.io/papers/chacha-AAAI2020.pdf)
> Traditional transportation approaches for traffic signal control can be categorized into following categories:
> - pre-timed control (Koonce and Rodegerdts 2008),
> - actuated control (Cools, Gershenson, and D’Hooghe 2013),
> - adaptive control (Lowrie 1990; Hunt et al. 1981) and
> - optimization based control (Varaiya 2013)

Aims to adress three key issues:
- **Scalability** for several thousand traffic lights
- **Coordination** between traffic signals at different intersections
- **Data feasibility**: Method should only use real-time data that is reasonably easy to obtain