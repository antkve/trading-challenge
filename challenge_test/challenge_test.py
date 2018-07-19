import os
from gym_seasonals.envs.remote_env import RemoteEnv
from gym_seasonals.envs.seasonals_env import random_agent

if __name__ == '__main__':
    env = RemoteEnv(os.environ['REMOTE_HOST'])
    random_agent(env)
    # your need to close the environment in order 
    # for the grader to submit your score correctly
    env.close() 
