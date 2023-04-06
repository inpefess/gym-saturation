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
# observation is a dictionary with two keys

print(observation.keys())

# %%
# ``real_obs`` value is a JSON representation of logic clauses

import pprint

from gym_saturation.constants import ACTION_MASK, REAL_OBS

pprint.pp(observation[REAL_OBS])

# %%
# ``action_mask`` is a ``numpy`` array of zeros and ones

print(type(observation[ACTION_MASK]))
print(observation[ACTION_MASK].shape)
print(observation[ACTION_MASK][:10])

# %%
# We can render the environment state in the TPTP format.
# By default, we are trying to prove a basic group theory lemma:
# every idempotent element equals the identity

env.render()

# %%
# here is an example of an episode during which we play random avail actions

terminated, truncated = False, False
while not (terminated or truncated):
    action = env.action_space.sample(mask=observation[ACTION_MASK])
    observation, reward, terminated, truncated, info = env.step(action)
env.close()

# %%
# the episode terminated with positive reward

print(terminated, truncated, reward)

# %%
# It means we arrived at a contradiction (``$false``) which proves the lemma.
# Notice the ``birth_step`` number of a contradiction, it shows how many steps
# we did to find proof.
pprint.pp(observation[REAL_OBS][-1])

# %%
# the package also provides a utility function for extracting only clauses
# which became parts of the proof (some steps might be unnecessary to find the
# proof)
from gym_saturation.utils import get_tstp_proof

print(get_tstp_proof(observation[REAL_OBS]))

# %%
# Age agent for iProver
# ----------------------
#
# We initialise iProver-based environment in the same way

env = gym.make("iProver-v0")

# %%
# Instead of a random agent, let's use Age agent which selects actions in the
# order they became available

observation, info = env.reset()
terminated, truncated = False, False
action = 0
while not (terminated or truncated):
    if observation[ACTION_MASK][action] == 1:
        observation, reward, terminated, truncated, info = env.step(action)
    action += 1
env.close()

# %%
# We still arrive at the contradiction but it might take a different number of
# steps. And the proof found looks a bit different

print(get_tstp_proof(observation[REAL_OBS]))
