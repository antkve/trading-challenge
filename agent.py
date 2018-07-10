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
        self.asset_ixs = range(len(asset_attributes['Return']))
        self.asset_histories = [[] for a in self.asset_ixs]
        self.null_action = np.zeros(len(asset_attributes['Return']))


# List of lists of dicts (phew)
# Each first-level list is an individual asset
# Each second-level list is a history of timesteps
# Each dict (timestep) has asset attributes as keys
    def __update_histories(self, assets_attributes):
        self.asset_histories = [asset_history + [timestep] 
                for asset_history, timestep 
                in zip(self.asset_histories, assets_attributes)]

    def act(self):
        action = self.null_action
        observation, reward, done, info = self.env.step(action)
        (index, day, can_trade, tradable_events, 
                lookahead_events_calendar, tradable_assets,
                asset_attributes, exposures, 
                portfolio_attributes, var_limit) = observation

        self.__update_histories(
                [{'Level':asset_attributes['Level'][a]
                    } for a in self.asset_ixs])
        return done

    def close(self):
        self.env.close()
        return [pandas.DataFrame(asset_history) 
                for asset_history in self.asset_histories]

def ema(ls, T):
    a = 2/(1 + T)
    emas = []
    last_ema = ls[0]
    for pt in ls:
        ema = ema + a * (pt - ema)

def cusum_filter(df, h, asset_attribute='Level'):
    df['ema'] = df.ewm(span=10).mean()
    print(df['ema'])
    S_pos = S_neg = 0
    filter_points = []
    last_ema = df['ema'][0]
    for ix, row in df.iterrows():
        S_pos = max(S_pos + row[asset_attribute] - last_ema, 0)
        S_neg = min(S_neg + row[asset_attribute] - last_ema, 0)
        if max(abs(S_neg), abs(S_pos)) > h:
            pt = row[asset_attribute]
            S_pos = S_neg = 0
        else: 
            pt = None
        filter_points.append(pt)
        last_ema = row['ema']
    df['filter_points'] = pandas.Series(filter_points)
    return df


def plot_df(df, x_col=None, assets=[0, 1], colours=None):
    print(df)
    x_col = x_col or df.index
    print(x_col)
    colours = colours or ['b' for colname in df.columns]
    for colname, colour in zip(df.columns, colours):
        if colname != x_col.name:
            plt.plot(x_col, 
                    df[colname], 
                    colour)
    plt.legend()
    plt.show()


def visualize_cusum(agent):
    done = False
    while not done:
        done = agent.act()
    dfs = agent.close()[5:]
    for ix in range(len(dfs)):
        dfs[ix]['cusum_sample'] = cusum_filter(dfs[ix], 10)['filter_points']
        plot_df(dfs[ix], colours=['b', 'y-', 'g^'])
           

env = gym.make('seasonals-v1')
visualize_cusum(Agent(env))
