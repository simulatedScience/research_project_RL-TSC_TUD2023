import os
import numpy as np
from common.metrics import Metrics
from environment import TSCEnv
from common.registry import Registry
from trainer.base_trainer import BaseTrainer
from agent.base import BaseAgent

@Registry.register_trainer("tsc")
class TSCTrainer(BaseTrainer):
    '''
    Register TSCTrainer for traffic signal control tasks.
    '''
    def __init__(
        self,
        logger,
        gpu=0,
        cpu=False,
        name="tsc"
    ):
        super().__init__(
            logger=logger,
            gpu=gpu,
            cpu=cpu,
            name=name
        )
        self.episodes = Registry.mapping['trainer_mapping']['setting'].param['episodes']
        self.steps = Registry.mapping['trainer_mapping']['setting'].param['steps']
        self.test_steps = Registry.mapping['trainer_mapping']['setting'].param['test_steps']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.action_interval = Registry.mapping['trainer_mapping']['setting'].param['action_interval']
        self.save_rate = Registry.mapping['logger_mapping']['setting'].param['save_rate']
        self.learning_start = Registry.mapping['trainer_mapping']['setting'].param['learning_start']
        self.update_model_rate = Registry.mapping['trainer_mapping']['setting'].param['update_model_rate']
        self.update_target_rate = Registry.mapping['trainer_mapping']['setting'].param['update_target_rate']
        self.test_when_train = Registry.mapping['trainer_mapping']['setting'].param['test_when_train']
        self.total_sensor_reads = self.steps / self.action_interval
        self.expected_throughput = 0
        # replay file is only valid in cityflow now. 
        # TODO: support SUMO and Openengine later
        
        # TODO: support other dataset in the future
        self.dataset = Registry.mapping['dataset_mapping'][Registry.mapping['command_mapping']['setting'].param['dataset']](
            os.path.join(Registry.mapping['logger_mapping']['path'].path,
                         Registry.mapping['logger_mapping']['setting'].param['data_dir'])
        )
        self.dataset.initiate(ep=self.episodes, step=self.steps, interval=self.action_interval)
        self.yellow_time = Registry.mapping['trainer_mapping']['setting'].param['yellow_length']
        # consists of path of output dir + log_dir + file handlers name
        self.log_file = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                     Registry.mapping['logger_mapping']['setting'].param['log_dir'],
                                     os.path.basename(self.logger.handlers[-1].baseFilename).rstrip('_BRF.log') + '_DTL.log'
                                     )

    def create_world(self):
        '''
        create_world
        Create world, currently support CityFlow World, SUMO World and Citypb World.

        :param: None
        :return: None
        '''
        # traffic setting is in the world mapping
        self.world = Registry.mapping['world_mapping'][Registry.mapping['command_mapping']['setting'].param['world']](
            self.path,
            Registry.mapping['command_mapping']['setting'].param['thread_num'],
            interface=Registry.mapping['command_mapping']['setting'].param['interface'])

    def create_metrics(self):
        '''
        create_metrics
        Create metrics to evaluate model performance, currently support reward, queue length, delay(approximate or real) and throughput.

        :param: None
        :return: None
        '''
        if Registry.mapping['command_mapping']['setting'].param['delay_type'] == 'apx':
            lane_metrics = ['rewards', 'queue', 'delay']
            world_metrics = ['real avg travel time', 'throughput']
        else:
            lane_metrics = ['rewards', 'queue']
            world_metrics = ['delay', 'real avg travel time', 'throughput']
        self.metric = Metrics(lane_metrics, world_metrics, self.world, self.agents)

    def create_agents(self):
        '''
        create_agents
        Create agents for traffic signal control tasks.

        :param: None
        :return: None
        '''
        self.agents: list[BaseAgent] = []
        agent: BaseAgent = Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, 0)
        num_agent = int(len(self.world.intersections) / agent.sub_agents)
        self.agents.append(agent)  # initialized N agents for traffic light control
        print(f"Created agent 1/{num_agent}:\n{self.agents[-1]}") # print first agent
        for i in range(1, num_agent):
            self.agents.append(Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, i))
            print(f"Created agent {i+1}/{num_agent}:\n{self.agents[-1]}")

        # for magd agents should share information 
        if Registry.mapping['model_mapping']['setting'].param['name'] == 'magd':
            for agent in self.agents:
                agent.link_agents(self.agents)

    def create_env(self):
        '''
        create_env
        Create simulation environment for communication with agents.

        :param: None
        :return: None
        '''
        # TODO: finalized list or non list
        self.env = TSCEnv(self.world, self.agents, self.metric)

    def train(self):
        '''
        train
        Train the agent(s).

        :param: None
        :return: None
        '''
        total_decision_num = 0
        flush = 0
        for episode in range(self.episodes):
            # TODO: check this reset agent
            self.metric.clear()
            self.total_sensor_reads = self.steps / self.action_interval #SJ: update total sensor reads for training
            # get initial observation
            last_obs = self.env.reset(
                    run_nbr=episode,
                    expected_throughput=self.expected_throughput,
                    total_sensor_reads=self.total_sensor_reads)  # agent * [sub_agent, feature]
            # reset agent of each intersection
            for a in self.agents:
                a.reset()
            if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
                # in cityflow environment, save replay file of episode
                if self.save_replay and episode % self.save_rate == 0:
                    self.env.eng.set_save_replay(True)
                    self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"episode_{episode}.txt"))
                else:
                    self.env.eng.set_save_replay(False)
            # start training one episode
            episode_loss = []
            i = 0
            while i < self.steps:
                # agents only act every `action_interval` time steps
                if i % self.action_interval == 0:
                    last_phase = np.stack([agent.get_phase() for agent in self.agents])  # [agent, intersections]

                    if total_decision_num > self.learning_start:
                        actions = []
                        for idx, agent in enumerate(self.agents):
                            actions.append(agent.get_action(last_obs[idx], last_phase[idx], test=False))                            
                        actions = np.stack(actions)  # [agent, intersections]
                    else:
                        actions = np.stack([agent.sample() for agent in self.agents])

                    actions_prob = []
                    for idx, agent in enumerate(self.agents):
                        actions_prob.append(agent.get_action_prob(last_obs[idx], last_phase[idx]))

                    rewards_list = []
                    for _ in range(self.action_interval):
                        obs, rewards, dones, _ = self.env.step(actions.flatten(), run_nbr=episode)
                        i += 1
                        rewards_list.append(np.stack(rewards))
                    rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                    self.metric.update(rewards)

                    cur_phase = np.stack([agent.get_phase() for agent in self.agents])
                    for idx, agent in enumerate(self.agents):
                        agent.remember(last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx], rewards[idx],
                            obs[idx], cur_phase[idx], dones[idx], f'{episode}_{i//self.action_interval}_{agent.id}')
                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0
                        # self.dataset.flush([agent.replay_buffer for agent in self.agents])
                    total_decision_num += 1
                    last_obs = obs
                # after we started learning, update the actor network every `self.update_model_rate` steps
                if total_decision_num > self.learning_start and\
                        total_decision_num % self.update_model_rate == self.update_model_rate - 1:

                    cur_loss_q = np.stack([agent.train() for agent in self.agents])  # TODO: training

                    episode_loss.append(cur_loss_q)
                # after we started learning, update the target network every `self.update_target_rate` steps
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                    [agent.update_target_network() for agent in self.agents]

                if all(dones):
                    break
            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0
            self.expected_throughput = self.metric.throughput() # update expected throughput to last measured throughput
            # log training status
            self.writeLog("TRAIN", episode, self.metric.real_average_travel_time(),\
                mean_loss, self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput())
            self.logger.info("step:{}/{}, q_loss:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(i, self.steps,\
                mean_loss, self.metric.rewards(), self.metric.queue(), self.metric.delay(), int(self.metric.throughput())))
            if episode % self.save_rate == 0:
                [agent.save_model(e=episode) for agent in self.agents]
            self.logger.info("episode:{}/{}, real avg travel time:{}".format(episode, self.episodes, self.metric.real_average_travel_time()))
            for j in range(len(self.world.intersections)):
                self.logger.debug("intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(j, self.metric.lane_rewards()[j],\
                     self.metric.lane_queue()[j]))
            if self.test_when_train:
                self.train_test(episode)
        # self.dataset.flush([agent.replay_buffer for agent in self.agents])
        [agent.save_model(e=self.episodes) for agent in self.agents]
        # SJ: added logging total number of model evaluations
        self.logger.info(f"Training completed using {total_decision_num} model evaluations.")

    def train_test(self, episode):
        '''
        train_test
        Evaluate model performance after each episode training process.

        :param e: number of episode
        :return self.metric.real_average_travel_time: travel time of vehicles
        '''
        self.total_sensor_reads = self.test_steps / self.action_interval # update total sensor reads for test
        obs = self.env.reset(
                run_nbr=episode,
                expected_throughput=self.expected_throughput,
                total_sensor_reads=self.total_sensor_reads)
        self.metric.clear()
        for a in self.agents:
            a.reset()
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([agent.get_phase() for agent in self.agents])
                actions = []
                for idx, agent in enumerate(self.agents):
                    actions.append(agent.get_action(obs[idx], phases[idx], test=True))
                actions = np.stack(actions)
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten(), run_nbr=episode)  # make sure action is [intersection]
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric.update(rewards)
            if all(dones):
                break
        self.expected_throughput = self.metric.throughput() # update expected throughput to last measured throughput
        # log testing process
        self.logger.info(f"Test step:{episode+1}/{self.episodes}, travel time :{self.metric.real_average_travel_time()}, "
                         + f"rewards:{self.metric.rewards()}, queue:{self.metric.queue()}, delay:{self.metric.delay()}, throughput:{int(self.metric.throughput())}"
        )
        self.writeLog("TEST", episode, self.metric.real_average_travel_time(),\
            100, self.metric.rewards(),self.metric.queue(),self.metric.delay(), self.metric.throughput())
        return self.metric.real_average_travel_time()

    def test(self, drop_load=True):
        '''
        test
        Test process. Evaluate model performance.

        :param drop_load: decide whether to load pretrained model's parameters
        :return self.metric: including queue length, throughput, delay and travel time
        '''
        self.expected_throughput = self.world.estimate_throughput()
        # print(f"testing with seed {self.seed}")
        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            if self.save_replay:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"final.txt"))
            else:
                self.env.eng.set_save_replay(False)
        self.metric.clear()
        if not drop_load:
            try:
                [agent.load_model(self.episodes) for agent in self.agents]
            except AttributeError as e:
                self.logger.error(f"No model to load.\n{e}")
        for agent in self.agents: # reload seed in agents
            agent.reload_noise_config()
        attention_mat_list = []
        model_evaluations: int = 0 # count how often each agents' models are evaluated
        self.total_sensor_reads = self.test_steps / self.action_interval # update total sensor reads for test
        obs = self.env.reset(
            run_nbr=0,
            expected_throughput=self.expected_throughput,
            total_sensor_reads=self.total_sensor_reads)
        for a in self.agents:
            a.reset()
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([agent.get_phase() for agent in self.agents])
                actions = []
                for idx, agent in enumerate(self.agents):
                    actions.append(agent.get_action(obs[idx], phases[idx], test=True))
                model_evaluations += 1
                actions = np.stack(actions)
                rewards_list = []
                for j in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten(), run_nbr=0)
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric.update(rewards)
            if all(dones):
                break
        self.logger.info(
            f"Final Travel Time is {self.metric.real_average_travel_time():.4f}, "
            + f"mean rewards: {self.metric.rewards():.4f}, "
            + f"queue: {self.metric.queue():.4f}, "
            + f"delay: {self.metric.delay():.4f}, "
            + f"throughput: {int(self.metric.throughput())}")
        return self.metric

    def writeLog(self, mode, step, travel_time, loss, cur_rwd, cur_queue, cur_delay, cur_throughput):
        '''
        writeLog
        Write log for record and debug.

        :param mode: "TRAIN" or "TEST"
        :param step: current step in simulation
        :param travel_time: current travel time
        :param loss: current loss
        :param cur_rwd: current reward
        :param cur_queue: current queue length
        :param cur_delay: current delay
        :param cur_throughput: current throughput
        :return: None
        '''
        res = Registry.mapping['model_mapping']['setting'].param['name'] + '\t' + mode + '\t' + str(
            step) + '\t' + "%.1f" % travel_time + '\t' + "%.1f" % loss + "\t" +\
            "%.2f" % cur_rwd + "\t" + "%.2f" % cur_queue + "\t" + "%.2f" % cur_delay + "\t" + "%d" % cur_throughput
        log_handle = open(self.log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()

