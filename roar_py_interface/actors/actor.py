import typing
import asyncio
import gymnasium as gym
from ..sensors.sensor import RoarPySensor
import time

"""
Helper functions to propose a observation dictionary key for a sensor and modify the dictionary of sensors
"""
def __propose_name_and_modify_dict(diction: dict, new_name: str, counter: int = 0):
    if counter < 1:
        if new_name + "_1" in diction:
            return __propose_name_and_modify_dict(diction, new_name, 2)
        else:
            diction[new_name + "_1"] = diction[new_name]
            diction.pop(new_name)
            return __propose_name_and_modify_dict(diction, new_name, 2)
    else:
        actual_name = new_name + "_" + str(counter)
        if actual_name in diction:
            return __propose_name_and_modify_dict(diction, new_name, counter + 1)
        else:
            return actual_name

"""
Helper functions to propose a observation dictionary key for a sensor and modify the dictionary of observation keys
"""
def __propose_name_and_modify_list(list_of_names: list, new_name: str, counter: int = 0):
    if counter < 1:
        if new_name + "_1" in list_of_names:
            return __propose_name_and_modify_list(list_of_names, new_name, 2)
        else:
            list_of_names.pop(list_of_names.index(new_name))
            list_of_names.append(new_name + "_1")
            return __propose_name_and_modify_list(list_of_names, new_name, 2)
    else:
        actual_name = new_name + "_" + str(counter)
        if actual_name in list_of_names:
            return __propose_name_and_modify_list(list_of_names, new_name, counter + 1)
        else:
            return actual_name

"""
Base Abstract class for all agents
Example control loop usage:

```
actor : RoarActor = init_actor() # get an agent from somewhere, probably from a world object
learning_agent = SACAgent(actor.get_action_spec(), actor.get_observation_spec()) # Initialize a learning agent using observation and action spec 

while True:
    observation = await agent.receive_observation()
    action = learning_agent.sample(observation)
    await actor.apply_action(action)
```
"""
class RoarPyActor:
    def __init__(
        self, 
        name: str,
        control_timestep : float = 0.05,
        force_real_control_timestep : bool = False,
    ):
        self.name = name
        self.control_timestep = control_timestep
        self.force_real_control_timestep = force_real_control_timestep
        self._last_action_t = 0.0

    """
    List all sensors on this actor
    """
    def get_sensors(self) -> typing.Iterable[RoarPySensor]:
        raise NotImplementedError()

    """
    Get action space specification for this actor
    """
    def get_action_spec(self) -> gym.Space:
        raise NotImplementedError()
    
    async def __apply_action(self, action: typing.Any) -> bool:
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def is_closed(self) -> bool:
        raise NotImplementedError()

    def __del__(self):
        try:
            self.close()
        finally:
            pass

    """
    Asynchronously apply action to this actor
    """
    async def apply_action(self, action: typing.Any) -> bool:
        
        if self.force_real_control_timestep:
            t = time.time()
            dt = t - self._last_action_t
            if dt < self.control_timestep:
                await asyncio.sleep(self.control_timestep - dt)
                self._last_action_t = self._last_action_t + self.control_timestep
            else:
                self._last_action_t = t
        return await self.__apply_action(action)

    """
    Get observation space specification for this actor

    This does not need to be inherited and implemented again because the actor 
    class constructs the observation space from the sensor list
    """
    def get_gym_observation_spec(self) -> gym.Space:
        spec_dict = {}
        for sensor in self.get_sensors():
            store_name = __propose_name_and_modify_dict(spec_dict, sensor.name)
            spec_dict[store_name] = sensor.get_gym_observation_spec()
        return gym.spaces.Dict(spec_dict)

    """
    Receive observation from all sensors on this actor

    This does not need to be inherited and implemented again because the actor
    class constructs the observation dictionary from the sensor list
    """
    async def receive_observation(self, observation: dict) -> dict[str, typing.Any]:
        observation_dict = {}
        obs_keys = []
        all_sensors = list(self.get_sensors())
        all_obs_coroutines : list[typing.Coroutine] = []
        
        for sensor in all_sensors:
            store_name = __propose_name_and_modify_list(obs_keys, sensor.name)
            obs_keys.append(store_name)
            all_obs_coroutines.append(sensor.receive_observation())

        all_received_obs = await asyncio.gather(*all_obs_coroutines)
        for i, obs in enumerate(all_received_obs):
            observation_dict[obs_keys[i]] = obs
        
        return observation_dict
    
    def get_last_observation(self) -> typing.Optional[dict[str,typing.Any]]:
        observation_dict = {}
        for sensor in self.get_sensors():
            store_name = __propose_name_and_modify_dict(observation_dict, sensor.name)
            sensor_lastobs = sensor.get_last_observation()
            if sensor_lastobs is not None:
                observation_dict[store_name] = sensor.get_last_observation()
            else:
                return None
        return observation_dict
    
    def get_last_gym_observation(self) -> typing.Optional[dict[str,typing.Any]]:
        observation_dict = {}
        for sensor in self.get_sensors():
            store_name = __propose_name_and_modify_dict(observation_dict, sensor.name)
            sensor_lastobs = sensor.get_last_gym_observation()
            if sensor_lastobs is not None:
                observation_dict[store_name] = sensor.get_last_gym_observation()
            else:
                return None
        return observation_dict

    def convert_obs_to_gym_obs(self, observation: dict[str,typing.Any]) -> dict[str,typing.Any]:
        obs_gym_dict = {}
        for sensor in self.get_sensors():
            store_name = __propose_name_and_modify_dict(obs_gym_dict, sensor.name)
            obs_gym_dict[store_name] = sensor.convert_obs_to_gym_obs(observation[store_name])
        return obs_gym_dict
    
class RoarPyResettableActor:
    async def __reset(self) -> None:
        raise NotImplementedError()

    """
    Reset this actor to its initial state
    If the actor is not resettable, this function should just return None
    """
    async def reset(self) -> None:
        is_base_class = isinstance(self, RoarPyActor)
        if is_base_class and self.force_real_control_timestep:
            t = time.time()
            dt = t - self._last_action_t
            if dt < self.control_timestep:
                await asyncio.sleep(self.control_timestep - dt)
                self._last_action_t = self._last_action_t + self.control_timestep
            else:
                self._last_action_t = t
        
        await self.__reset()