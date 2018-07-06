from gym_seasonals.envs.seasonals_env import SeasonalsEnv
from gym_seasonals.envs.seasonals_env import random_agent
import gym

class Agent:
    def __init__(self, env):
        self.env = env
        observation, info = env.reset()
        (index, day, can_trade, tradable_events, lookahead_events_calendar, tradable_assets,
                asset_attributes, exposures, portfolio_attributes, var_limit) = observation
        self.n_assets = len(asset_attributes['Return'])

    def act(self, observation):
        (index, day, can_trade, tradable_events, lookahead_events_calendar, tradable_assets,
                asset_attributes, exposures, portfolio_attributes, var_limit) = observation
        print("index: {}\n day: {}\n can_trade: {}\n tradable_events: {}\n lookahead_events_calendar: {}\n tradable_assets: {}\n asset_attributes: {}\n exposures: {}\n portfolio_attributes: {}\n var_limit: {}".format(index, day, can_trade, tradable_events, lookahead_events_calendar, tradable_assets, asset_attributes, exposures, portfolio_attributes, var_limit))

        if can_trade == 1:

            target_action = np.round(env.action_space.sample() * tradable_assets * 0.01, -5).astype(int)

            var_utilised = portfolio_var(exposures + target_action, asset_returns_history)
            action = target_action if var_utilised < 0.99 * var_limit else null_action
    

env = gym.make('seasonals-v1')
env.close()
