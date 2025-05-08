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

NUM_EPISODES = 1 # number of episodes to simulate
for episode in range(1, NUM_EPISODES+1): 
    state = env.reset() # reset env to start of level
    done = False
    total_reward = 0

    while not done: # play episode until finished
        # action of agent (insert policy here)
        action = env.action_space.sample()

        # run one time step of the env using the agent action
        new_state, reward, done, info = env.step(action)
        total_reward += reward
        env.render() # render environment time step
    print(f"Episode #{episode} || Total reward: {total_reward}")

print(f"\n\n{state.shape},\n {reward},\n {done},\n {info}") # print info

env.close()

# Resize obsvervation wrapper
# Frame stack wrapper
# GrayScale wrapper