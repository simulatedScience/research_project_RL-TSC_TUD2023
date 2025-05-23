import gymnasium as gym
import numpy as np
from agent.base import BaseAgent


class TSCEnv(gym.Env):
    """
    Environment for Traffic Signal Control task.
    Parameters
    ----------
    world: World object
    agents: list of agents, corresponding to each intersection in world.intersections
    metric: Metric object, used to calculate evaluation metric
    """

    def __init__(self, world, agents, metric):
        """
        :param world: one world object to interact with agents. Support multi world
        objects in different TSCEnvs.
        :param agents: single agents, each control all intersections. Or multi agents,
        each control one intersection.
        actions is a list of actions, agents is a list of agents.
        :param metric: metrics to evaluate policy.
        """
        self.world = world
        self.eng = self.world.eng
        self.n_agents: int = len(agents) * agents[0].sub_agents
        # test agents number == intersection number
        assert len(world.intersection_ids) == self.n_agents
        self.agents: list[BaseAgent] = agents
        action_dims = [agent.action_space.n * agent.sub_agents for agent in agents]
        # total action space of all agents.
        self.action_space = gym.spaces.MultiDiscrete(action_dims)
        self.metric = metric

    def step(self, actions, run_nbr=0):
        """
        :param actions: keep action as N_agents * 1
        """
        if not actions.shape:
            assert(self.n_agents == 1)
            actions = actions[np.newaxis]
        else:
            assert len(actions) == self.n_agents
        self.world.step(actions)

        if not len(self.agents) == 1:
            obs = [agent.get_ob(run_nbr=run_nbr) for agent in self.agents]
            # obs = np.expand_dims(np.array(obs),axis=1)
            rewards = [agent.get_reward() for agent in self.agents]
            # rewards = np.expand_dims(np.array(rewards),axis=1)
        else:
            obs = [self.agents[0].get_ob(run_nbr=run_nbr)]
            rewards = [self.agents[0].get_reward()]
        dones = [False] * self.n_agents
        # infos = {"metric": self.metric.update()}
        infos = {}

        return obs, rewards, dones, infos

    def reset(self, run_nbr=0, expected_throughput=1000, total_sensor_reads=360):
        self.world.reset(expected_throughput=expected_throughput, total_sensor_reads=total_sensor_reads)
        if not len(self.agents) == 1:
            obs = [agent.get_ob(run_nbr=run_nbr) for agent in self.agents]  # [agent, sub_agent==1, feature]
            # obs = np.expand_dims(np.array(obs),axis=1)
        else:
            obs = [self.agents[0].get_ob(run_nbr=run_nbr)]  # [agent==1, sub_agent, feature]
        return obs
