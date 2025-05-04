import gym
from collections import defaultdict
import numpy as np

class  MarioAgent:
    """Initializes Mario agent"""
    def __init__(self,
                 env: gym.Env,
                 learning_rate: float,
                 initial_epsilon: float,
                 epsilon_decay: float,
                 final_epsilon: float,
                 discount_factor: float):
        self.env = env
        self.q_values = defaultdict(np.array(lambda: np.zeros(env.action_space.n)))
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
