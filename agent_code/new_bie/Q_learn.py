import events as e
import numpy as np
import os
from collections import deque
import pickle
import random

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 5  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Custom Events
VALID_ACTION = "VALID_ACTION"
MOVED_IN_CYCLE = "MOVED_IN_CYCLE"

MOVED_TO_COIN = "MOVED_TO_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"

MOVED_TO_CRATE = "MOVED_TO_CRATE"
MOVED_AWAY_FROM_CRATE = "MOVED_AWAY_FROM_CRATE"

MEANINGFUL_BOMB_DROP = "MEANINGFUL_BOMB_DROP"
NOT_MEANINGFUL_BOMB_DROP = "NOT_MEANINGFUL_BOMB_DROP"

SAFE_FROM_BOMB = "SAFE_FROM_BOMB"
NOT_SAFE_FROM_BOMB = "NOT_SAFE_FROM_BOMB"

NOT_KILLED_SELF = "NOT_KILLED_SELF"

class QLearning:
    def __init__(self, train=True, epsilon=0.1, alpha=0.5, gamma=0.9, model_path=None, reward_log=False):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.train = train

        self.q_table = {}  # Q-value table
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
        self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

        self.model_path = model_path
        if not train:
            self.load_model(model_path)

        self.reward_log = reward_log
        if reward_log is not False:
            self.reward_log_list = []
            self.current_round_reward = 0


    def act(self, game_state):
        state_key = self.take_game_key(game_state)
        q_values = self.q_table.get(state_key, {a: random.uniform(-2, 0) for a in self.actions})
        self.q_table.setdefault(state_key, q_values)

        best_action = max(q_values, key=q_values.get)

        if random.random() < self.epsilon:
            action = random.choice(list(set(self.actions) - {best_action}))
        else:
            action = best_action

        return action


    def log(self, reward):
        if self.reward_log is not False:
            self.reward_log_list.append(reward)

            _path = os.getcwd()
            _path = os.path.abspath(os.path.join(_path, '../..'))
            reward_log_path = os.path.join(_path, self.reward_log) 

            f = open(reward_log_path, 'w')
            f.write("episode,reward\n")
            for i, r in enumerate(self.reward_log_list):
                f.write(f"{i},{r}\n")


    def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
        if self.train:
            self.update_q_table(old_game_state, self_action, new_game_state, events)


    def end_of_round(self, last_game_state, last_action, events):
        if self.train:
            self.update_q_table(last_game_state, last_action, last_game_state, events)
            self.save_model(self.model_path)
        
        if self.reward_log is not False:
            self.current_round_reward += self.reward_from_events(events)
            self.log(self.current_round_reward)
            self.current_round_reward = 0


    def update_q_table(self, old_state, action, new_state, events):
        old_key = self.take_game_key(old_state)
        new_key = self.take_game_key(new_state)

        reward = self.reward_from_events(events)

        self.q_table.setdefault(old_key, {a: random.uniform(-2, 0) for a in self.actions})
        self.q_table.setdefault(new_key, {a: random.uniform(-2, 0) for a in self.actions})

        max_future_q = max(self.q_table[new_key].values())
        current_q = self.q_table[old_key][action]

        self.q_table[old_key][action] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

        self.transitions.append({
            'old_state': old_state,
            'action': action,
            'new_state': new_state,
            'events': events
        })

        if self.reward_log is not False:
            self.current_round_reward += reward


    # def 


    def reward_from_events(self, events):
        rewards = {
            VALID_ACTION: 1,
            e.INVALID_ACTION: -100,
            MOVED_IN_CYCLE: -150,

            e.COIN_COLLECTED: 300,
            MOVED_TO_COIN: 10,
            MOVED_AWAY_FROM_COIN: -10,

            e.CRATE_DESTROYED: 100,
            MOVED_TO_CRATE: 5,      # maybe not the best idea to reward this 
            MOVED_AWAY_FROM_CRATE: -5,  # maybe not the best idea to reward this

            e.KILLED_OPPONENT: 500,
            e.OPPONENT_ELIMINATED: 100,

            e.KILLED_SELF: -1000,
            NOT_KILLED_SELF: 20,

            e.GOT_KILLED: -1000,
            e.SURVIVED_ROUND: 50,

            MEANINGFUL_BOMB_DROP: 100,
            NOT_MEANINGFUL_BOMB_DROP: -50,

            SAFE_FROM_BOMB: 50,
            NOT_SAFE_FROM_BOMB: -100
        }

        return sum(rewards.get(event, 0) for event in set(events))


    def save_model(self, filename):
        _path = os.getcwd()
        _path = os.path.abspath(os.path.join(_path, '../..'))
        path = os.path.join(_path, self.model_path)

        f = open(path, 'wb')
        pickle.dump(self.q_table, f)


    def load_model(self, filename):
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            print("[!] Q-table file not found. Starting fresh.")
            self.q_table = {}


    def take_game_key(self, game_state):
        features = self.__game_state_to_features(game_state)
        return self.__hashing_game_state(features)


    def __hashing_game_state(self, game_state):
        field = tuple(map(tuple, game_state['field']))
        explosion_map = tuple(map(tuple, game_state['explosion_map']))
        self_info = (
            game_state['self'][1],
            game_state['self'][2],
            tuple(game_state['self'][3])
        )
        bombs = tuple((tuple(pos), timer) for pos, timer in game_state['bombs'])
        coins = tuple(tuple(pos) for pos in game_state['coins'])
        others = tuple(
            (
                game_state['others'][i][1],
                game_state['others'][i][2],
                tuple(game_state['others'][i][3])
            ) for i in range(len(game_state['others']))
        )
        return (field, explosion_map, self_info, bombs, coins, others)


    def __game_state_to_features(self, game_state):
        x, y = game_state['self'][3]
        h, w = game_state['field'].shape
        offset = 5
        game_state['field'] = game_state['field'][max(0, x - offset):min(w, x + offset + 1), max(0, y - offset):min(h, y + offset + 1)]
        game_state['explosion_map'] = game_state['explosion_map'][max(0, x - offset):min(w, x + offset + 1), max(0, y - offset):min(h, y + offset + 1)]
        return game_state
