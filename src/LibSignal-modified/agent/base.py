from common.registry import Registry


@Registry.register_model('base')
class BaseAgent(object):
    '''
    BaseAgent Class is mainly used for creating a base agent and base methods.
    '''
    def __init__(self, world):
        # revise if it is multi-agents in one model
        self.world = world
        self.seed = Registry.mapping['command_mapping']['setting'].param['seed']
        self.sub_agents = 1
        self.FAILURE_CHANCE = Registry.mapping['command_mapping']['setting'].param['failure_chance']
        self.TPR = Registry.mapping['command_mapping']['setting'].param['tpr']
        self.FPR = Registry.mapping['command_mapping']['setting'].param['fpr']
    
    def reload_noise_config(self):
        """
        Reload noise parameters from Registry.
        """
        self.FAILURE_CHANCE = Registry.mapping['command_mapping']['setting'].param['failure_chance']
        self.TPR = Registry.mapping['command_mapping']['setting'].param['tpr']
        self.FPR = Registry.mapping['command_mapping']['setting'].param['fpr']
        self.seed = Registry.mapping['command_mapping']['setting'].param['seed']

    def get_ob(self):
        raise NotImplementedError()

    def get_reward(self):
        raise NotImplementedError()

    def get_action(self, ob, phase):
        raise NotImplementedError()

    def get_action_prob(self, ob, phase):
        return None
