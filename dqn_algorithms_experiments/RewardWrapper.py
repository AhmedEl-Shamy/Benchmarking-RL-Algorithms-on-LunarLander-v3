import gymnasium as gym
import numpy as np

class DistancePenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty_weight=0.1):
        super().__init__(env)
        self.penalty_weight = penalty_weight

    def reward(self, reward):
        x, y = self.env.unwrapped.state[0], self.env.unwrapped.state[1]
        distance = np.sqrt(x**2 + y**2)
        return reward - self.penalty_weight * distance