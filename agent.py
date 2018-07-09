from gym_seasonals.envs.seasonals_env import SeasonalsEnv
from gym_seasonals.envs.seasonals_env import portfolio_var
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas

class Agent:

    def __init__(self, env):
        self.env = env
        (index, day, can_trade, tradable_events, 
                lookahead_events_calendar, tradable_assets,
                asset_attributes, exposures, 
                portfolio_attributes, var_limit) = env.reset()
        self.assets_history = {'Level':[]}
        n_assets = len(asset_attributes['Return'])
        self.null_action = np.zeros(n_assets)

    def __update_history(self, attributes):
        for key, attribute in attributes.items():
            self.assets_history[key].append(attribute)

    def act(self):
        action = self.null_action
        observation, reward, done, info = env.step(action)
        (index, day, can_trade, tradable_events, 
                lookahead_events_calendar, tradable_assets,
                asset_attributes, exposures, 
                portfolio_attributes, var_limit) = observation
        self.__update_history({'Level':asset_attributes['Level']})
        return done

    def close(self):
        self.env.close()
        return pandas.DataFrame(self.assets_history)


def visualize(agent):
    done = False
    while not done:
        done = agent.act()
    plot1_y = [pt[0] for pt in agent.assets_history['Level']]
    plot2_y = [pt[1] for pt in agent.assets_history['Level']]
    plot_x = range(len(agent.assets_history['Level']))
    plt.plot(plot_x, plot1_y, 'r')
    plt.plot(plot_x, plot2_y, 'b')
    plt.show()

           

env = gym.make('seasonals-v1')
visualize(Agent(env))
visualize(Agent(env))



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



    
