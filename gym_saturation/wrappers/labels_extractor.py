"""Labels extractor wrapper."""

import re

import gymnasium as gym


class LabelsExtractor(gym.ObservationWrapper):
    """
    Labels extractor wrapper.

    >>> from gym_saturation.envs.vampire_env import VampireEnv
    >>> env = LabelsExtractor(VampireEnv())
    >>> observation, info = env.reset()
    >>> type(observation)
    <class 'dict'>
    >>> observation.keys()
    dict_keys(['labels', 'observation'])
    >>> type(observation["labels"])
    <class 'tuple'>
    >>> type(observation["labels"][0])
    <class 'str'>
    """

    def observation(
        self, observation: tuple[str, ...]
    ) -> dict[str, tuple[str, ...]]:
        """
        Return a modified observation.

        :param observation: The observation
        :returns: The modified observation
        """
        return {
            "labels": tuple(
                re.findall(r"cnf\((\w+),.+\)\.", clause)[0]
                for clause in observation
            ),
            "observation": observation,
        }
