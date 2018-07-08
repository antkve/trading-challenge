from gym_seasonals.envs.seasonals_env import SeasonalsEnv
from gym_seasonals.envs.seasonals_env import portfolio_var
import gym
import matplotlib.pyplot as plt
import numpy as np

class VisAgent:
    def __init__(self, env):
        self.env = env
        res = env.reset()
        print(len(res))
        observation  = res
        (index, day, can_trade, tradable_events, 
                lookahead_events_calendar, tradable_assets,
                asset_attributes, exposures, 
                portfolio_attributes, var_limit) = observation
        n_assets = len(asset_attributes['Return'])
        self.asset_returns_history = np.zeros((0, n_assets))
        self.assets_price_history = np.zeros((0, n_assets))
        self.action = self.null_action = np.zeros(n_assets)



    def act(self):
        observation, reward, done, info = \
                self.env.step(self.action)
        (index, day, can_trade, tradable_events, 
                lookahead_events_calendar, tradable_assets,
                asset_attributes, exposures, 
                portfolio_attributes, var_limit) = observation
        self.asset_returns = asset_attributes['Return']
        self.asset_returns_history = np.append(
                self.asset_returns_history, 
                self.asset_returns.reshape(1, -1), axis=0)
        self.assets_price_history = np.append(
                self.assets_price_history,
                asset_attributes['Level'].reshape(1, -1), 
                axis=0)
        if can_trade == 1:
            target_action = np.round(self.env.action_space.sample() 
                    * tradable_assets * 0.01, -5).astype(int)
            var_utilised = portfolio_var(exposures 
                    + target_action, self.asset_returns_history)
            self.action = target_action \
                    if var_utilised < 0.99 * var_limit \
                    else self.null_action
        else: self.action = self.null_action
        if done: return True
        else: return False
    
    def close(self):
        self.env.close()
        plot1_y = [pt[0] for pt in self.assets_price_history]
        plot2_y = [pt[1] for pt in self.assets_price_history]
        plot_x = range(len(self.assets_price_history))
        plt.plot(plot_x, plot1_y, 'r')
        plt.plot(plot_x, plot2_y, 'b')
        plt.show()
   

env = gym.make('seasonals-v1')
agent = VisAgent(env)
done = False
while not done:
    done = agent.act()
agent.close() 
