import gymnasium as gym
import numpy as np

class DistancePenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty_weight=0.1):
        super().__init__(env)
        self.penalty_weight = penalty_weight

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x, y = obs[0], obs[1]
        distance = np.sqrt(x**2 + y**2)
        shaped_reward = reward - self.penalty_weight * distance
        return obs, shaped_reward, terminated, truncated, info