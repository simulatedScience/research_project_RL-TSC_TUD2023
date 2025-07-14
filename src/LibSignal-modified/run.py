import task
import trainer
import agent
import dataset
from common.registry import Registry
from common import interface
from common.utils import *
from utils.logger import *
import time
from datetime import datetime
import argparse



def parse_args():
    # parseargs
    parser = argparse.ArgumentParser(description='Run Experiment')
    parser.add_argument('--thread_num', type=int, default=12, help='number of threads')  # used in cityflow
    parser.add_argument('--ngpu', type=str, default="1", help='gpu to be used')  # choose gpu card
    parser.add_argument('--prefix', type=str, default='test', help="the number of prefix in this running process")
    parser.add_argument('--seed', type=int, default=None, help="seed for pytorch backend")
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--interface', type=str, default="libsumo", choices=['libsumo','traci'], help="interface type") # libsumo(fast) or traci(slow)
    parser.add_argument('--delay_type', type=str, default="apx", choices=['apx','real'], help="method of calculating delay") # apx(approximate) or real

    parser.add_argument('-t', '--task', type=str, default="tsc", help="task type to run")
    # parser.add_argument('-a', '--agent', type=str, default="dqn", help="agent type of agents in RL environment")
    parser.add_argument('-a', '--agent', type=str, default="presslight", help="agent type of agents in RL environment") # SJ: switched agent default to presslight
    parser.add_argument('-w', '--world', type=str, default="sumo", choices=['cityflow','sumo'], help="simulator type") # SJ: switched world default to sumo
    parser.add_argument('--failure_chance', type=float, default=0.0, help="failure chance of sensors")
    parser.add_argument('--tpr', type=float, default=0.0, help="chance of adding noise to NN inputs")
    parser.add_argument('--fpr', type=float, default=0.15, help=r"noise range for NN inputs. 95% of noise will be within this range")
    # parser.add_argument('-n', '--network', type=str, default="cityflow1x1", help="network name")
    parser.add_argument('-n', '--network', type=str, default="sumo1x1", help="network name")
    parser.add_argument('-d', '--dataset', type=str, default='onfly', help='type of dataset in training process')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu
    print(f"GPU used: {args.ngpu}")

    # logging_level = logging.INFO
    # if args.debug:
    #     logging_level = logging.DEBUG

    return args

class Runner:
    def __init__(self, pArgs):
        """
        instantiate runner object with processed config and register config into Registry class
        """
        self.config, self.duplicate_config = build_config(pArgs)
        self.config_registry()

    def config_registry(self):
        """
        Register config into Registry class
        """

        interface.Command_Setting_Interface(self.config)
        interface.Logger_param_Interface(self.config)  # register logger path
        interface.World_param_Interface(self.config)
        if self.config['model'].get('graphic', False):
            param = Registry.mapping['world_mapping']['setting'].param
            if self.config['command']['world'] in ['cityflow', 'sumo']:
                roadnet_path = param['dir'] + param['roadnetFile']
            else:
                roadnet_path = param['road_file_addr']
            interface.Graph_World_Interface(roadnet_path)  # register graphic parameters in Registry class
        interface.Logger_path_Interface(self.config)
        # make output dir if not exist
        if not os.path.exists(Registry.mapping['logger_mapping']['path'].path):
            os.makedirs(Registry.mapping['logger_mapping']['path'].path)        
        interface.Trainer_param_Interface(self.config)
        interface.ModelAgent_param_Interface(self.config)

    def run(self,
            filename_addon: str = "",
            ):
        logging_level = logging.INFO
        if args.debug:
            logging_level = logging.DEBUG

        logger = setup_logging(logging_level, filename_addon=filename_addon)

        self.trainer = Registry.mapping['trainer_mapping']\
            [Registry.mapping['command_mapping']['setting'].param['task']](logger)
        self.task = Registry.mapping['task_mapping']\
            [Registry.mapping['command_mapping']['setting'].param['task']](self.trainer)

        failure_chance = Registry.mapping['command_mapping']['setting'].param['failure_chance']
        tpr = Registry.mapping['command_mapping']['setting'].param['tpr']
        fpr = Registry.mapping['command_mapping']['setting'].param['fpr']
        run_id = Registry.mapping['command_mapping']['setting'].param['seed']
        self.trainer.load_seed_from_config()
        run_identifier = f"id={run_id}_fc={failure_chance}_tpr={tpr}_fpr={fpr}"
        logger.info(
            f"Running RL Experiment: {Registry.mapping['command_mapping']['setting'].param['prefix']} " + \
            f"\nrun_identifier: {run_identifier}" if run_id != "" else ""
        )
        # save training info into a md file
        save_run_info()
        start_time = time.time()
        self.task.run(drop_load=False)
        logger.info(f"Total time taken: {time.time() - start_time}")

def save_run_info():
    """
    Save run info into a md file. Info to save is loaded from config files registered in Registry class
    """
    # load simulator config file
    sumo_config: str = os.path.join('configs/sim', Registry.mapping['command_mapping']['setting'].param['network'] + '.cfg')
    with open(sumo_config) as file:
        sumo_dict = json.load(file)
    dataset: str = sumo_dict['combined_file'] # dataset file path
    # load agent config
    dic_agent_conf: dict[str, int] = Registry.mapping['model_mapping']['setting']
    try:
        layer_size: int = dic_agent_conf.param['d_dense']
    except KeyError:
        layer_size: int = None
    
    run_info = {
        'agent_type': Registry.mapping['command_mapping']['setting'].param['agent'],
        'road_network': Registry.mapping['command_mapping']['setting'].param['network'],
        'dataset': dataset,
        'n_epsiodes': Registry.mapping['trainer_mapping']['setting'].param['episodes'],
        'layers': layer_size, # single dense hidden layer size
        'failure_chance': Registry.mapping['command_mapping']['setting'].param['failure_chance'],
        'tpr': Registry.mapping['command_mapping']['setting'].param['tpr'],
        'fpr': Registry.mapping['command_mapping']['setting'].param['fpr'],
        'seed': Registry.mapping['command_mapping']['setting'].param['seed'],
    }
    run_info_text = f"""
- training started at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
- trained on data: `{run_info['dataset']}`
- trained `{run_info['n_epsiodes']}` episodes
- independent DQN per intersection with layers: `{run_info['layers']}`
- sensor failure simulation updated Feb 2024
- failure chance: `{run_info['failure_chance']}`
- TPR: `{run_info['tpr']}`
- FPR: `{run_info['fpr']}`
- random seed: `{run_info['seed']}`
"""
    # experiment folder
    file_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_folder = Registry.mapping['logger_mapping']['path'].path
    os.makedirs(os.path.join(exp_folder, 'run_info'), exist_ok=True)
    info_file_path = os.path.join(exp_folder, 'run_info', f'{file_datetime}_run_info.md')
    with open(info_file_path, 'w') as file:
        file.write(run_info_text)
    print(f"Run info saved at: {info_file_path}")



if __name__ == '__main__':
    import random
    # use common.utils.load_config to load yml config files
    # args = argparse.Namespace(
    #     thread_num = 8,
    #     ngpu = 1,
    #     prefix = "exp_1",
    #     seed = None,
    #     debug = True,
    #     interface = "libsumo",
    #     delay_type = "apx",

    #     task = "tsc",
    #     agent = "colight",
    #     world = "sumo",
    #     network = "sumo1x1",
    #     dataset = "onfly",
    # )
    args = parse_args()
    seed = random.randint(0, 1000)
    args = argparse.Namespace(
        thread_num = 22, # use 8 CPU threads
        ngpu = 1, # use 1 GPU
        prefix = f"exp_14072025_disturbed_seed{seed}_eps50_nn128", # exp3_2_undisturbed_100
        seed = seed, # random seed
        debug = True,
        interface = "libsumo", # use (lib)sumo for simulation
        delay_type = "apx", # approximated delay

        task = "tsc",
        agent = "presslight", # frap, presslight, colight, fixedtime
        world = "sumo",
        network = "sumo1x3_synth_uniform", # sumo1x5_atlanta, sumo1x1, sumo1x1_colight, sumo1x3
        dataset = "onfly",

        # failure_chance = 0., # 0 / .1   # failure_chance,
        # tpr = 1., # 1 / .8   # true positive rate,
        # fpr = 0., # 0 / .15   # false positive rate,
        failure_chance = 0.1, # 0 / .1   # failure_chance,
        tpr = 0.8, # 1 / .8   # true positive rate,
        fpr = 0.15, # 0 / .15   # false positive rate,
    )
    test = Runner(args)
    # train with moderate sensor failure rate
    test.run()
    # play short beep sound when done
    import winsound
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 300

    winsound.Beep(frequency, duration)

