import hashlib

import numpy as np

from . import BaseGenerator
from world import world_cityflow, world_sumo #, world_openengine


class LaneVehicleGenerator(BaseGenerator):
    '''
    Generate state or reward based on statistics of lane vehicles.

    :param world: World object
    :param I: Intersection object
    :param fns: list of statistics to get, currently support "lane_count", "lane_waiting_count" , "lane_waiting_time_count", "lane_delay", "lane_pressure" and "pressure". 
        "lane_count": get number of running vehicles on each lane. 
        "lane_waiting_count": get number of waiting vehicles(speed less than 0.1m/s) on each lane. 
        "lane_waiting_time_count": get the sum of waiting time of vehicles on the lane since their last action. 
        "lane_delay": the delay of each lane: 1 - lane_avg_speed/speed_limit.
        "lane_pressure": the number of vehicles that in the in_lane minus number of vehicles that in out_lane.
        "pressure": difference of vehicle density between the in-coming lane and the out-going lane.

    :param in_only: boolean, whether to compute incoming lanes only. 
    :param average: None or str, None means no averaging, 
        "road" means take average of lanes on each road, 
        "all" means take average of all lanes.
    :param negative: boolean, whether return negative values (mostly for Reward).
    :param FAILURE_CHANCE: float, chance of sensor failure (0.0 - 1.0). default 0.0 (no disturbance)
    :param TPR: float, true positive rate for sensor detections (0.0 - 1.0). default 1.0 (no disturbance)
    :param FPR: float, false positive rate for sensor detections (0.0 - 1.0). default 0.0 (no disturbance)
    :param seed: int, random seed for consistent sensor failures. default None
    '''
    def __init__(self,
                 world,
                 intersection,
                 fns,
                 in_only=False,
                 average=None,
                 negative=False,
                 FAILURE_CHANCE=0.0,
                 TPR=1.0,
                 FPR=0.0,
                 seed=None
                 ):
        self.world = world
        self.intersection = intersection
        self.seed = seed
        if self.seed is None:
            self.seed = np.random.randint(0, 2**16 - 1)
        
        self.FAILURE_CHANCE = FAILURE_CHANCE
        self.TPR = TPR # true positive rate
        self.FPR = FPR # false positive rate

        # get lanes of intersections
        self.lanes = []
        if in_only:
            roads = intersection.in_roads
        else:
            roads = intersection.roads

        # ---------------------------------------------------------------------
        # # resort roads order to NESW
        # if self.I.lane_order_cf != None or self.I.lane_order_sumo != None:
        #     tmp = []
        #     if isinstance(world, world_sumo.World):
        #         for x in ['N', 'E', 'S', 'W']:
        #             if self.I.lane_order_sumo[x] != -1:
        #                 tmp.append(roads[self.I.lane_order_sumo[x]])
        #             # else:
        #             #     tmp.append('padding_roads')
        #         roads = tmp

        #         # TODO padding roads into 12 dims
        #         for r in roads:
        #             if not self.world.RIGHT:
        #                 tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(ob[-1]), reverse=True)
        #             else:
        #                 tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(ob[-1]))
        #             self.lanes.append(tmp)

        #     elif isinstance(world, world_cityflow.World):
        #         for x in ['N', 'E', 'S', 'W']:
        #             if self.I.lane_order_cf[x] != -1:
        #                 tmp.append(roads[self.I.lane_order_cf[x]])
        #             # else:
        #             #     tmp.append('padding_roads')
        #         roads = tmp

        #         # TODO padding roads into 12 dims
        #         for road in roads:
        #             from_zero = (road["startIntersection"] == I.id) if self.world.RIGHT else (road["endIntersection"] == I.id)
        #             self.lanes.append([road["id"] + "_" + str(i) for i in range(len(road["lanes"]))[::(1 if from_zero else -1)]])

        #     else:
        #         raise Exception('NOT IMPLEMENTED YET')
        
        # else:
        #     if isinstance(world, world_sumo.World):
        #         for r in roads:
        #             if not self.world.RIGHT:
        #                 tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(ob[-1]), reverse=True)
        #             else:
        #                 tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(ob[-1]))
        #             self.lanes.append(tmp)
        #             # TODO: rank lanes by lane ranking [0,1,2], assume we only have one digit for ranking
        #     elif isinstance(world, world_cityflow.World):
        #         for road in roads:
        #             from_zero = (road["startIntersection"] == I.id) if self.world.RIGHT else (road["endIntersection"] == I.id)
        #             self.lanes.append([road["id"] + "_" + str(i) for i in range(len(road["lanes"]))[::(1 if from_zero else -1)]])
            
        #     else:
        #         raise Exception('NOT IMPLEMENTED YET')


            

        # ---------------------------------------------------------------------------------------------------------------
        # TODO: register it in Registry
        if isinstance(world, world_sumo.World):
            for r in roads:
                if not self.world.RIGHT:
                    tmp = sorted(intersection.road_lane_mapping[r], key=lambda ob: int(ob[-1]), reverse=True)
                else:
                    tmp = sorted(intersection.road_lane_mapping[r], key=lambda ob: int(ob[-1]))
                self.lanes.append(tmp)
                # TODO: rank lanes by lane ranking [0,1,2], assume we only have one digit for ranking
        elif isinstance(world, world_cityflow.World):
            for road in roads:
                from_zero = (road["startIntersection"] == intersection.id) if self.world.RIGHT else (road["endIntersection"] == intersection.id)
                self.lanes.append([road["id"] + "_" + str(i) for i in range(len(road["lanes"]))[::(1 if from_zero else -1)]])
        # ---------------------------------------------------------------------------------------------------------------
        
        # elif isinstance(world, world_openengine.World):
        #     for r in roads:
        #         if self.world.RIGHT:
        #             tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(str(ob)[-1]), reverse=True)
        #         else:
        #             tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(str(ob)[-1]))
        #         self.lanes.append(tmp)
        else:
            raise Exception('NOT IMPLEMENTED YET')

        # subscribe functions
        self.world.subscribe(fns)
        self.fns = fns

        # calculate result dimensions
        size = sum(len(x) for x in self.lanes)
        if average == "road":
            size = len(roads)
        elif average == "all":
            size = 1
        self.ob_length = len(fns) * size
        if self.ob_length == 3:
            self.ob_length = 4

        self.average = average
        self.negative = negative

    def generate(self, pad_to: int = 4, run_nbr=0) -> np.ndarray:
        '''
        generate
        Generate state or reward based on current simulation state.
        
        :param: pad_to: int, pad the result with zeros to this length if not averaged.
        :return ret (np.ndarray): state or reward
        '''
        results = [self.world.get_info(fn) for fn in self.fns]
        #need modification here
        ret = np.array([])
        for i, fn in enumerate(self.fns):
            result = results[i]

            # pressure returns result of each intersections, so return directly
            if self.intersection.id in result:
                ret = np.append(ret, result[self.intersection.id])
                continue
            fn_result = np.array([])

            for road_lanes in self.lanes:
                road_result = []
                for lane_id in road_lanes:
                    # add disturbance to the queue lengths/ vehicle counts, when not averaging
                    # averaging indicates the function is used for reward calculation, which should not be disturbed
                    if self.average is None and fn in ("lane_count", "lane_waiting_count"):
                        # simulate sensor failure, broken sensors remain broken are entire episode
                        if self.FAILURE_CHANCE > 0 and deterministic_random(lane_id, self.seed + run_nbr) < self.FAILURE_CHANCE:
                            result[lane_id] = 0
                        else:
                            # simulate noisy sensor readings
                            # simulate missed detections when vehicles were actually there using true positive rate
                            result[lane_id] = simulate_true_positives(result[lane_id], self.TPR)
                            # simulate misdetections when vehicles were not actually there using false positive rate
                            result[lane_id] += simulate_false_positives(self.world.expected_throughput, self.world.total_sensor_reads, self.FPR, self.TPR)
                    road_result.append(result[lane_id])
                if self.average == "road" or self.average == "all":
                    road_result = np.mean(road_result)
                else:
                    road_result = np.array(road_result)
                fn_result = np.append(fn_result, road_result)
            
            if self.average == "all": # SJ: calculate average of all lanes to get a single number
                fn_result = np.mean(fn_result)
            ret = np.append(ret, fn_result)
        if self.negative:
            ret = ret * (-1)
        # origin_ret = ret # SJ: was unused
        if len(ret) > 1 and len(ret) < pad_to:
            ret = np.pad(ret, (0, pad_to - len(ret)))
        # SJ: pad returned list to 4 if not averaged. Why not use np.pad?
        # if len(ret) == 3:
        #     ret_list = list(ret)
        #     ret_list.append(0)
        #     ret = np.array(ret_list)
        # if len(ret) == 2:
        #     ret_list = list(ret)
        #     ret_list.append(0)
        #     ret_list.append(0)
        #     ret = np.array(ret_list)
        return ret

def simulate_true_positives(n_samples: int, tpr: float = 0.60) -> int:
    """
    Simulate some of the vehicles that are actually there not being detected by the sensor using a Binomial distribution.
    Each vehicle is detected with probability `tpr`.
    
    Args:
        n_samples (int): The number of vehicles to simulate.
        tpr (float): The true positive rate (probability of a vehicle being detected).
    
    Returns:
        (int) The number of vehicles that were detected (always >= 0 and <= `n_samples`)
    """
    # Simulate N Bernoulli trials
    detections = np.random.binomial(1, tpr, n_samples)
    return detections.sum()

def simulate_false_positives(vehicles_per_hour: float = 2800, sensor_reads_per_hour: float = 360, fpr: float = 0.65, tpr: float=1.0) -> int:
    """
    Simulate misdetections of vehicles that are not actually there using a Poisson distribution.
    To achieve a false positive rate of `fpr` over the entire simulation, we calculate the mean number of false positives per timestep from the number of sensor reads and expected throughput.
    V = throughput per hour
    R = sensor reads per hour
    FPR = false positive rate
    
    True vehicle detection mean: V/R
    Mean number of false positives per timestep: V/R * FPR -> = mean of Poisson distribution
    Mean detections per timestep with FPR: V/R * (1 + FPR)

    Args:
        vehicles_per_hour (float): The expected throughput of vehicles per hour.
        sensor_reads_per_hour (float): The number of sensor reads per hour.
        fpr (float): The false positive rate (false positives / total detections).

    Returns:
        int: Number of false positives in this timestep.
    """
    # mean_false_positives = vehicles_per_hour / sensor_reads_per_hour * fpr
    mean_false_positives = fpr*tpr*vehicles_per_hour / (1-fpr) / sensor_reads_per_hour
    false_positives = np.random.poisson(mean_false_positives)
    return false_positives


def deterministic_random(input_data, original_seed, shape=None):
    """
    Generate a random number based on the input data and the original seed. This function is deterministic, i.e. the same input data and original seed will always produce the same random number, changing either one will change the random number.

    Args:
        input_data: The input data to be hashed and used to generate the random number.
        original_seed: The original seed used to generate the random number.
        
    """
    # Hash the input data to create a seed
    input_hash = hashlib.sha256(str(input_data).encode()).hexdigest()
    input_seed = int(input_hash, 16) % (2**32 - 1)  # Reduce hash to fit within uint32
    
    # Combine the original seed and the input seed
    combined_seed = original_seed ^ input_seed

    # Create a new random generator with the combined seed
    rng = np.random.default_rng(combined_seed)

    # Generate random numbers
    return rng.random(shape)


if __name__ == "__main__":
    from world.world_cityflow import World
    world = World("examples/configs.json", thread_num=1)
    laneVehicle = LaneVehicleGenerator(world, world.intersections[0], ["count"], False, "road")
    for _ in range(100):
        world.step()
    print(laneVehicle.generate())
