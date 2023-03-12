
import time

from cnn import PyTorchDQNAgent
from utils import play_game, play_game2
from game_environment import Snake, SnakeNumpy
import numpy as np

import json

version = 'v17.1'
with open('model_config/{:s}.json'.format(version), 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames'] # keep frames >= 2
    max_time_limit = m['max_time_limit']
    supervised = bool(m['supervised'])
    n_actions = m['n_actions']
    obstacles = bool(m['obstacles'])
    buffer_size = m['buffer_size']

epsilon, epsilon_end = 1, 0.01
reward_type = 'current'
sample_actions = False
n_games_training = 8 * 16
decay = 0.97
games = 512
episodes = 2 * (10**5)
log_frequency = 500
games_eval = 8

env = SnakeNumpy(board_size=board_size, frames=frames,
            max_time_limit=max_time_limit, games=games,
            frame_mode=True, obstacles=obstacles, version=version)

agent = PyTorchDQNAgent(board_size=board_size, frames=frames,
                        n_actions=n_actions, use_target_net=True)

ct = time.time()
_ = play_game2(env, agent, n_actions, n_games=games, record=True,
               epsilon=epsilon, verbose=True, reset_seed=False,
               frame_mode=True, total_frames=games*64)
print('Playing {:d} frames took {:.2f}s'.format(games*64, time.time()-ct))

env = SnakeNumpy(board_size=board_size, frames=frames,
            max_time_limit=max_time_limit, games=n_games_training,
            frame_mode=True, obstacles=obstacles, version=version)
env2 = SnakeNumpy(board_size=board_size, frames=frames,
            max_time_limit=max_time_limit, games=games_eval,
            frame_mode=True, obstacles=obstacles, version=version)

# create a new instance of PyTorchDQNAgent


# set the number of episodes and steps per episode to train for
num_episodes = 1000
steps_per_episode = 100

# loop over the episodes
num_episodes = 1000
for episode in range(num_episodes):
    board = env.reset()
    legal_moves = env.get_legal_moves()

    done = False
    while not done:
        action = agent.act(board, legal_moves)
        next_board, reward, done, _ = env.step(action)
        agent.update(board, action, reward, next_board, done, legal_moves)
        board = next_board
    agent.train()
