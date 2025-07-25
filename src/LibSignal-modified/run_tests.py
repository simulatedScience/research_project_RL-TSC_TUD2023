# import task
# import trainer
# import agent
# import dataset
from common.registry import Registry
from common import interface
from common.utils import *
from utils.logger import *
import time
from datetime import datetime
import argparse
from run_train import save_run_info

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
            failure_chances: list = [0.0],
            tprs: list = [0.0],
            fprs: list = [0.15],
            num_repetitions: int = 1,
            min_id: int = 0,
            ):
        logging_level = logging.INFO
        if args.debug:
            logging_level = logging.DEBUG
            
        logger = setup_logging(logging_level, filename_addon=filename_addon)
        
        self.trainer = Registry.mapping['trainer_mapping']\
            [Registry.mapping['command_mapping']['setting'].param['task']](logger)
        self.task = Registry.mapping['task_mapping']\
            [Registry.mapping['command_mapping']['setting'].param['task']](self.trainer)
        
        for fpr in fprs:
            for failure_chance in failure_chances:
                for tpr in tprs:
                    first_model = True
                    for run_id in range(min_id, min_id + num_repetitions):
                        run_seed: int = np.random.randint(0, 100000)
                        Registry.mapping['command_mapping']['setting'].param['failure_chance'] = failure_chance
                        Registry.mapping['command_mapping']['setting'].param['tpr'] = tpr
                        Registry.mapping['command_mapping']['setting'].param['fpr'] = fpr
                        Registry.mapping['command_mapping']['setting'].param['seed'] = run_seed
                        self.trainer.load_seed_from_config()
                        run_identifier = f"id={run_id}_seed={run_seed}_fc={failure_chance}_tpr={tpr}_fpr={fpr}"
                        logger.info(
                            f"Running RL Experiment: {Registry.mapping['command_mapping']['setting'].param['prefix']} " + \
                            f"\nrun_identifier: {run_identifier}" if run_id != "" else ""
                        )
                        save_run_info()
                        start_time = time.time()
                        self.task.run(drop_load = not first_model)
                        logger.info(f"Total time taken: {time.time() - start_time}")
                        first_model = False


if __name__ == '__main__':
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
    # num_repetitions = 20
    # for tpr in [0.0, 1.0]:
    #     for fpr in [0.15]:
    #         for failure_chance in [0.0, 0.05, 0.1, 0.15]:
    #             for run_id in range(num_repetitions):
    args = parse_args()
    new_args = argparse.Namespace(
        thread_num = 2,
        ngpu = 0,
        # prefix = "exp6_disturbed_seed100_eps30_nn32", # exp3_1_undisturbed_100
        # prefix = "exp6_1_maxpressure", # exp3_1_undisturbed_100
        # prefix = "error:to_be_replaced", # exp3_1_undisturbed_100
        # prefix = "exp4_0_undisturbed_synth_100",
        # prefix = "exp_14072025_disturbed_seed489_eps50_nn128",
        # prefix = "exp_new_undisturbed_seed907_eps30_nn128",
        seed = 0,
        debug = True,
        interface = "libsumo",
        delay_type = "apx",

        task = "tsc",
        agent = "presslight", # frap, presslight, colight, fixedtime, maxpressure
        world = "sumo",
        network = "sumo1x3", # sumo1x5_atlanta, sumo1x1, sumo1x1_colight, sumo1x3
        dataset = "onfly",
        
        failure_chance = 0.15, # failure_chance,
        tpr = 0.6, # true positive rate,
        fpr = 0.65, # false positive rate,
    )
    args.__dict__.update(new_args.__dict__)
    test = Runner(args)
    start_time = time.time()
    # single test run
    # test.run(
    #     failure_chances=[0.], # 0.1
    #     tprs=[1.0], # 0.8
    #     fprs=[0.65], # 0.3
    #     num_repetitions=1,
    # )
    # tests 5-6 (4*4*4*8=64*8=512 runs)
    # test.run(
    #     failure_chances=[0.15, 0.1, 0.05, 0.0],
    #     tprs=[0.6, 0.8, 0.95, 1.0],
    #     fprs=[0.65, 0.3, 0.15, 0.0],
    #     num_repetitions=16,
    # )
    test.run( # for quick testing
        failure_chances=[0.15, 0.1, 0.05, 0.0],
        tprs=[0.6, 0.8, 0.95, 1.0],
        fprs=[0.65, 0.3, 0.15, 0.0],
        num_repetitions=360,
        min_id=0,
    )
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time}")
    # play short beep sound when done
    # import winsound
    # frequency = 2500  # Set Frequency To 2500 Hertz
    # duration = 300
    # winsound.Beep(frequency, duration)

    # tests 3-4 (6*6*5*15=180*15=2700 runs)
    # test.run(
    #     failure_chances=[0.2, 0.15, 0.1, 0.05, 0.0],
    #     tprs=[0.6, 0.8, 0.85, 0.9, 0.95, 1.0],
    #     fprs=[0.65, 0.3, 0.2, 0.15, 0.1, 0.0],
    #     num_repetitions=15,
    # )
    # test.run(
    #     failure_chances=[0.0],
    #     tprs=[0.0, 1.0],
    #     fprs=[1.0, 0.0],
    #     num_repetitions=1,
    # )
