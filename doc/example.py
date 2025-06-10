# Random agent for Vampire
# We can make a prover environment as any other Gymnasium one.

# We will also add a wrapper to extract formulae labels.


import gymnasium as gym

from gym_saturation.wrappers import LabelsExtractor

env = LabelsExtractor(gym.make("Vampire-v0"))


# before using the environment, we should reset it


observation, info = env.reset()


# ~gym-saturation~ environments don't return any ~info~


print(info)


# Observation is a tuple of CNF formulae.

# By default, we are trying to prove a basic group theory lemma: every idempotent element equals the identity.


print("Observation:")
print("\n".join(observation["observation"]))


# Wrappers extracts formulae labels for us:


labels = list(observation["labels"])
print(labels)


# Here is an example of an episode during which we play random actions.
# We set the random seed for reproducibility.


import random

random.seed(0)

terminated, truncated = False, False
while not (terminated or truncated):
    action = random.choice(labels)
    observation, reward, terminated, truncated, info = env.step(action)
    print("Action:", action, "Observation:")
    print("\n".join(observation["observation"]))
    labels.remove(action)
    labels += list(observation["labels"])

env.close()


# the episode is terminated


print(terminated, truncated)


# It means we arrived at a contradiction (~$false~) which proves the lemma.


print(observation["observation"][-1])
# Age agent for iProver
# We initialise iProver-based environment in the same way as Vampire-based one:


env = LabelsExtractor(gym.make("iProver-v0"))


# To run in Jupyter

import nest_asyncio

nest_asyncio.apply()


# Instead of a random agent, let's use Age agent which selects actions in the order they appear


observation, info = env.reset()
print("Observation:")
print("\n".join(observation["observation"]))
labels = list(observation["labels"])
terminated = False
while not terminated:
    action = labels.pop(0)
    observation, reward, terminated, truncated, info = env.step(action)
    print("Action:", action, "Observation:")
    print("\n".join(observation["observation"]))
    labels += list(observation["labels"])
env.close()


# We still arrive at a contradiction


print(terminated, truncated)
print(observation["observation"][-1])
