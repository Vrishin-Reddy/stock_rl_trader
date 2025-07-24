import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, df, forecast_value):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.forecast_value = forecast_value
        self.current_step = 0
        self.balance = 10000
        self.position = 0  # 0 = no position, 1 = holding stock

        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation: current price, forecast, position (can be extended)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)

    def _get_obs(self):
        price = self.df.loc[self.current_step, 'Close']
        return np.array([price, self.forecast_value, self.position], dtype=np.float32)

    def step(self, action):
        done = False
        reward = 0
        price = self.df.loc[self.current_step, 'Close']

        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.buy_price = price
        elif action == 2 and self.position == 1:  # Sell
            reward = price - self.buy_price
            self.position = 0

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        obs = self._get_obs()
        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.balance = 10000
        return self._get_obs()
