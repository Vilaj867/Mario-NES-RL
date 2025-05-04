from torch import nn
import numpy as np

from gym.wrappers import FrameStack
from gym.wrappers import GrayScaleObservation

# nes emu
from nes_py.wrappers import JoypadSpace

# SMB environment for OpenAI Gym 
import gym_super_mario_bros

# create SMB environment (World 1-1)
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
print(f"\n\n{env.spec}\n") # see spec info of env
env = JoypadSpace(env, [['right'],['right', 'B'],['right', 'A', 'B']]) # limits the movement states of agent

done = True # if mario dies or completes level

# simulate running game for 5000 time steps
for step in range(5000):
    if done: 
        state = env.reset() # reset env to start of level

    # action of agent (insert policy here)
    action = env.action_space.sample()

    # run one time step of the env using the agent action
    state, reward, done, info = env.step(action)

    env.render() # render environment time step

# For testing purposes
print(f"\n\n{state.shape},\n {reward},\n {done},\n {info}") # print info
env = GrayScaleObservation(env, keep_dim=True) # apply gray scale wrapper to environment to reduce info
state, reward, done, info = env.step(action=0)
print(f"\n\n{state.shape}")
env.close()
