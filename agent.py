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

        self.__update_history(
                {
            'Level':asset_attributes['Level']
            })
        return done

    def close(self):
        self.env.close()
        return pandas.DataFrame(self.assets_history)


def cusum_filter(srs, h):
    df = pandas.DataFrame(srs)
    df['ema'] = df.ewm(span=10)
    S_pos = S_neg = 0
    filter_points = []
    last_ema = df['ema'][0]
    for ix, row in df.iterrows():
        S_pos = max(S_pos + row[srs.name] - last_ema, 0)
        S_neg = min(S_pos + row[srs.name] - last_ema, 0)
        pt = row[price] \
                if max(abs(S_neg), abs(S_pos)) > h \
                else None
        filter_points.append(pt)
        last_ema = row['ema']
    return pandas.Series(filter_points)


def plot_df(df, x_col=None, assets=[0, 1], colours=None):
    x_col = x_col or df.index.name
    colours = colours or [['b' 
            for asset in assets] 
        for colname in df.columns]
    for colname, colourset in zip(df.columns, colours):
        if colname != x_col:
            col = df[colname].apply(pandas.Series)
            for asset, colour in zip(assets, colourset):
                plt.plot(df[x_col], col[asset], colour)
    plt.legend()
    plt.show()


def visualize(agent):
    done = False
    while not done:
        done = agent.act()
    df = agent.close()
    print(df)
    df['cusum_sample'] = cusum_filter(df.Level, 0.0001)
    plot_df(df, colours=[['r', 'b'], ['gt', 'yt']])
           

env = gym.make('seasonals-v1')
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
