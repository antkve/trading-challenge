from gym_seasonals.envs.seasonals_env import SeasonalsEnv
from gym_seasonals.envs.seasonals_env import portfolio_var
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas

class Agent:

    def __init__(self, *attributes):
        self.assets_history = {key:[] for key in attributes}
        self.null_action = np.zeros(n_assets)

    def __update_history(self, attributes):
        for key, attribute in attributes.items():
            self.assets_history[key].append(attribute)

    def act(self, out):
        observation, reward, done, info = out
        (index, day, can_trade, tradable_events, 
                lookahead_events_calendar, tradable_assets,
                asset_attributes, exposures, 
                portfolio_attributes, var_limit) = observation
        self.__update_history(
                {**asset_attributes, 
                    'exposures':exposures, 
                    **portfolio_attributes})
        return self.null_action
            
    def close(self):
        self.env.close()
        return pandas.DataFrame(self.assets_history)

env = gym.make('seasonals-v1')
agent = Agent('Level', 'TradeCost', 'index')
action = agent.act(env.reset())
while not done:
    out = env.step(action)
    action = agent.act(out)
agent.close()



class RandomAgent(Agent):
    def __init__(self, *args, **kwargs ):
        super().__init__(self, *args, **kwargs)

    def act(self, out):
        observation, reward, done, info = out
        (index, day, can_trade, tradable_events, 
                lookahead_events_calendar, tradable_assets,
                asset_attributes, exposures, 
                portfolio_attributes, var_limit) = observation
        if can_trade == 1:
            target_action = np.round(self.env.action_space.sample() 
                    * tradable_assets * 0.01, -5).astype(int)
            var_utilised = portfolio_var(exposures 
                    + target_action, self.assets_history['Return'])
            self.action = target_action \
                    if var_utilised < 0.99 * var_limit \
                    else self.null_action
        
        else: self.action = self.null_action
        self.__update_history(
                {**asset_attributes, 
                    'exposures':exposures, 
                    **portfolio_attributes})
        return action, done



    
def visualize(agent, env, **kwargs):
    while not done:
        out = 
        done = agent.act(out)
