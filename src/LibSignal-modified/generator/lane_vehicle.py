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
    '''
    def __init__(self,
                 world,
                 I,
                 fns,
                 in_only=False,
                 average=None,
                 negative=False,
                 FAILURE_CHANCE=0.0,
                 NOISE_CHANCE=0.0,
                 NOISE_RANGE=0.15,
                 seed=None
                 ):
        self.world = world
        self.I = I
        self.seed = seed
        
        self.FAILURE_CHANCE = FAILURE_CHANCE
        self.NOISE_CHANCE = NOISE_CHANCE
        self.NOISE_RANGE = NOISE_RANGE

        # get lanes of intersections
        self.lanes = []
        if in_only:
            roads = I.in_roads
        else:
            roads = I.roads

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
                    tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(ob[-1]), reverse=True)
                else:
                    tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(ob[-1]))
                self.lanes.append(tmp)
                # TODO: rank lanes by lane ranking [0,1,2], assume we only have one digit for ranking
        elif isinstance(world, world_cityflow.World):
            for road in roads:
                from_zero = (road["startIntersection"] == I.id) if self.world.RIGHT else (road["endIntersection"] == I.id)
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

    def generate(self, pad_to: int = 4) -> np.ndarray:
        '''
        generate
        Generate state or reward based on current simulation state.
        
        :param: pad_to: int, pad the result with zeros to this length if not averaged.
        :return ret (np.ndarray): state or reward
        '''
        results = [self.world.get_info(fn) for fn in self.fns]
        #need modification here
        ret = np.array([])
        for i in range(len(self.fns)):
            result = results[i]

            # pressure returns result of each intersections, so return directly
            if self.I.id in result:
                ret = np.append(ret, result[self.I.id])
                continue
            fn_result = np.array([])

            for road_lanes in self.lanes:
                road_result = []
                for lane_id in road_lanes:
                    # SJ: add disturbance to the results
                    # SJ: laneid required for deterministic random
                    if self.average is None:
                        if deterministic_random(lane_id, self.seed) < self.FAILURE_CHANCE:
                            result[lane_id] = 0
                        elif np.random.random() < self.NOISE_CHANCE:
                            result[lane_id] = modify_with_relative_error(result[lane_id], range_value=self.NOISE_RANGE)
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
    
def sample_discrete_gaussian(mean, std_dev):
    """
    Simplified sampling from the discrete Gaussian distribution using numpy's normal distribution and rounding.
    
    Args:
        mean (float): The mean of the discrete Gaussian distribution.
        std_dev (float): The standard deviation of the discrete Gaussian distribution.
    
    Returns:
        (int) A sample from the discrete Gaussian distribution.
    """
    return int(np.round(np.random.normal(mean, std_dev)))

def modify_with_relative_error(n: int, range_value: float = 0.15) -> int:
    """
    Modify the given number `n` by introducing a relative error using Gaussian noise.
    
    Args:
        n (int): The number to be modified.
        range_value (float): 2*standard deviation of the Gaussian noise = 95% of the values will be in the range [n-range_value, n+range_value].
    
    Returns:
        (int) The modified value of `n` with the introduced relative error.
    """
    std_dev = (range_value / 2) * n  # scale the standard deviation based on the original value
    modified_value = n + sample_discrete_gaussian(0, std_dev)  # Add noise centered around 0
    return modified_value

def deterministic_random(input_data, original_seed, shape=None):
    """
    Generate a random number based on the input data and the original seed. This function is deterministic, i.e. the same input data and original seed will always produce the same random number, changing either one will change the random number.

    Args:
        input_data (_type_): _description_
        original_seed (_type_): _description_
        shape (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
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



