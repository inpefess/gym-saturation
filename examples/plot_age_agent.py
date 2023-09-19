"""
Random and age agents for Vampire and iProver
==============================================
"""
# %%
# Random agent for Vampire
# -------------------------
#
# To make a ``gym-saturation`` environment, we have to import the package

import gymnasium as gym

import gym_saturation

# %%
# then we can make a prover environment as any other Gymnasium one

env = gym.make("Vampire-v0")

# %%
# before using the environment, we should reset it

observation, info = env.reset()

# %%
# ``gym-saturation`` environments don't return any ``info``

print(info)

# %%
# observation is a tuple of JSON representations of logic clauses

import pprint

pprint.pp(observation)

# %%
# We can render the environment state in the TPTP format.
# By default, we are trying to prove a basic group theory lemma:
# every idempotent element equals the identity

env.render()

# %%
# here is an example of an episode during which we play random actions.
# We set the random seed for reproducibility.

env.action_space.seed(0)
terminated, truncated = False, False
while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
env.close()

# %%
# the episode terminated with positive reward

print(terminated, truncated, reward)

# %%
# It means we arrived at a contradiction (``$false``) which proves the lemma.
# Notice the ``birth_step`` number of a contradiction, it shows how many steps
# we did to find proof.
pprint.pp(observation[-1])

# %%
# Age agent for iProver
# ----------------------
#
# We initialise iProver-based environment in the same way

env = gym.make("iProver-v0")

# %%
# Instead of a random agent, let's use Age agent which selects actions in the
# order they appear

observation, info = env.reset()
terminated, truncated = False, False
action = 0
while not (terminated or truncated):
    observation, reward, terminated, truncated, info = env.step(action)
    action += 1
env.close()

# %%
# We still arrive at contradiction but it takes a different number of steps

print(terminated, truncated, reward)
pprint.pp(observation[-1])
