import gym
import numpy as np
import rl_game

class GameEnv(gym.Env):
    def __init__(self):
        self.env = rl_game.Environment()

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return np.array(obs.v, dtype=np.float32), {}

    def step(self, action):
        result = self.env.step(action)

        obs = np.array(result.obs.v, dtype=np.float32)

        return obs, result.reward, result.done, False, {}