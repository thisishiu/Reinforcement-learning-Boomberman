import events as e
import numpy as np
import os
from collections import deque
import pickle
import random

np.set_printoptions(linewidth=np.inf)

TRANSITION_HISTORY_SIZE = 5

# Asset
WALL = -1
FREE = 0
CRATE = 1

class QLearning:
    def __init__(self, train=True, epsilon=0.1, alpha=0.5, gamma=0.9, model_path=None, reward_log=None):
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
        if reward_log is not None:
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

        # return 'BOMB'
        return action


    def log(self):
        if self.reward_log is not False:
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
            # self.save_model(self.model_path)
            # print(f"[!] Q-table saved to {self.model_path}")
        
        if self.reward_log is not None:
            self.current_round_reward += self.reward_from_events(events)
            self.reward_log_list.append(self.current_round_reward)
            self.current_round_reward = 0
            # print(f"[!] Reward log saved to {self.reward_log}")

    
    def end(self, last_game_state, last_action, events):
        if self.train:
            self.update_q_table(last_game_state, last_action, last_game_state, events)
            self.save_model()
            
        if self.reward_log is not None:
            self.current_round_reward += self.reward_from_events(events)
            self.reward_log_list.append(self.current_round_reward)
            self.log()
    

    def update_q_table(self, old_state, action, new_state, events):
        old_key = self.take_game_key(old_state)
        new_key = self.take_game_key(new_state)
        
        # Check if the action is valid
        events = self.check_move(old_state, action, new_state, events)
        reward = self.reward_from_events(events)
        print(f"[!] Events: {events}")
        print(f"[!] Action: {action}")
        print(f"[!] Reward: {reward}")
        

        if self.q_table.get(old_key) is None:
            # Initialize Q-values for the old state if not present
            self.q_table[old_key] = {a: random.uniform(-2, 0) for a in self.actions}
        if self.q_table.get(new_key) is None:
            # Initialize Q-values for the new state if not present
            self.q_table[new_key] = {a: random.uniform(-2, 0) for a in self.actions}

        max_future_q = max(self.q_table[new_key].values())
        current_q = self.q_table[old_key][action]

        self.q_table[old_key][action] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

        self.transitions.append(
            self.__game_state_to_features(new_state)
        )

        if self.reward_log is not None:
            self.current_round_reward += reward
        
        # print(f"[!] Events: {events}")
        # print(f"[!] Action: {action}")
        # print(f"[!] Position: {new_state['self'][3]}")
        # print(f"[!] Reward: {reward}")
        # print("-------------------------------------------")


    def reward_from_events(self, events):
        game_rewards = {
            e.INVALID_ACTION: -100, #
            e.RUN_IN_LOOP:-50,      #
            e.WAITED: -10,          
            e.GET_TRAPPED: -50,

            e.GOT_KILLED: -150,
            e.KILLED_SELF: -300,
            e.SURVIVED_ROUND: 3,

            e.MOVED_UP:-1,
            e.MOVED_RIGHT: -1,
            e.MOVED_DOWN: -1,
            e.MOVED_LEFT: -1,
            
            e.COIN_COLLECTED: 100,
            e.COIN_DISTANCE_REDUCED: 10,
            e.COIN_DISTANCE_INCREASED: -5,
            e.COIN_FOUND: 15, 
            
            e.BOMB_DROPPED: -10,
            e.BOMB_DROPPED_NEXT_TO_CRATE: 12,
            e.BOMB_DROPPED_NEXT_TO_OPPONENT: 15,
            e.BOMB_AVOIDED : 30,
            e.BOMB_DISTANCE_INCREASED: 15,
            
            e.CRATE_DESTROYED: 20,
            e.CRATE_DISTANCE_REDUCED: 7,
            e.CRATE_DISTANCE_INCREASED: -2,
            e.CRATE_FOUND: 2, 
            e.CRATE_WITHOUT_DROPPING_BOMB: -15,

            e.KILLED_OPPONENT: 500,
            e.OPPONENT_DISTANCE_REDUCED: 10,
            e.OPPONENT_DISTANCE_INCREASED: -5,
            e.OPPONENT_WITHOUT_DROPPING_BOMB: -15
        }

        return sum(game_rewards.get(event, 0) for event in set(events))


    def check_move(self, old_game_state, self_action, new_game_state, events):
        subevent = []

        old_state = self.__game_state_to_features(old_game_state)
        new_state = self.__game_state_to_features(new_game_state)

        # Check if the action is valid
        if self_action not in self.actions:
            subevent.append(e.INVALID_ACTION)


        # Check if the agent moved in a cycle
        range_x = [float('inf'), 0]
        range_y = [float('inf'), 0]
        if len(self.transitions) > 4:
            for tr in self.transitions:
                if tr['pos'][0] < range_x[0]:
                    range_x[0] = tr['pos'][0]
                if tr['pos'][0] > range_x[1]:
                    range_x[1] = tr['pos'][0]
                if tr['pos'][1] < range_y[0]:
                    range_y[0] = tr['pos'][1]
                if tr['pos'][1] > range_y[1]:
                    range_y[1] = tr['pos'][1]
        if (range_x[1] - range_x[0] < 3)  and (range_y[1] == range_y[0]) or (range_y[1] - range_y[0] < 3) and (range_x[1] == range_x[0]):
            subevent.append(e.MOVED_IN_CYCLE)
        
        # Check if the agent moved to a coin
        if len(new_state['distance_to_coins']) > 0 and len(old_state['distance_to_coins']) and new_state['distance_to_coins'][0] < old_state['distance_to_coins'][0]:
            subevent.append(e.MOVED_TO_COIN)
        elif len(new_state['distance_to_coins']) > 0 and len(old_state['distance_to_coins']) and new_state['distance_to_coins'][0] > old_state['distance_to_coins'][0]:
            subevent.append(e.MOVED_AWAY_FROM_COIN)
            
        # check if the agent dodged a bomb
        if len(new_state['explosion_map']) > 0 and new_state['pos'] in new_state['explosion_map']:
            subevent.append(e.NOT_SAFE_FROM_BOMB)
        elif len(new_state['explosion_map']) > 0 and new_state['pos'] not in new_state['explosion_map']:
            subevent.append(e.SAFE_FROM_BOMB)
        
        # check if the agent dropped a bomb meaningfully
        if self_action == 'BOMB':
            # check if the bomb will explode on a crate or an opponent
            if any(old_game_state['field'][x, y] in [CRATE] for x, y in new_state['explosion_map']) or any((new_state['vector_to_others'][i][0] == 0 and new_state['vector_to_others'][i][1] == 0) and self.__l2_distance(new_state['vector_to_others'][i]) < 5 for i in range(len(new_state['vector_to_others']))):
                subevent.append(e.MEANINGFUL_BOMB_DROP)
            else:
                subevent.append(e.NOT_MEANINGFUL_BOMB_DROP)
                
        # check if the agent kill enemy
        if e.KILLED_OPPONENT in events:
            subevent.append(e.KILLED_OPPONENT)
        if e.OPPONENT_ELIMINATED in events:
            subevent.append(e.OPPONENT_ELIMINATED)
                
        events = list(set(events + subevent))
        
        # check if the agent killed itself
        if e.NOT_SAFE_FROM_BOMB in events and e.WAITED in events:
            subevent.append(e.WAITING_IN_DANGER_ZONE)

        return events
        
    def __able_to_bomb(self, game_state):
        ...    
    
    
    def __l2_distance(self, pos:list):
        """
        Calculate the L2 distance from the agent's position to a given position.
        """
        x, y = pos
        return np.sqrt(x**2 + y**2)
        

    def __vector_to_others(self, self_, others):
        vec = []
        for other in others:
            offset = np.array(other[3]) - np.array(self_[3])
            vec.append(offset)
        return vec
    
    
    def __tile_to_explosion(self, bombs, field):
        explosion_map = []
        w, h = field.shape
        for bomb in bombs:
            # horizontal check
            x, y = bomb[0]
            if field[y, x+1] != WALL:
                explosion_map.append((x+1, y))
                if field[y, min(x+2, w-1)] != WALL:
                    explosion_map.append((min(x+2, w-1), y))
            if field[y, x-1] != WALL:
                explosion_map.append((x-1, y))
                if field[y, max(x-2, 0)] != WALL:
                    explosion_map.append((x-2, y))
            
            # vertical check
            if field[y+1, x] != WALL:
                explosion_map.append((x, y+1))
                if field[min(y+2, h-1), x] != WALL:
                    explosion_map.append((x, min(y+2, h-1)))
            if field[y-1, x] != WALL:
                explosion_map.append((x, y-1))
                if field[max(y-2, 0), x] != WALL:
                    explosion_map.append((x, max(y-2, 0)))
            explosion_map.append((x, y))  # Add the bomb's position itself

        # print(f"[!] Explosion map: {explosion_map}")
        return explosion_map


    def __distance_to_coins(self, self_, coins):
        vec = []
        for coin in coins:
            offset = np.array(coin) - np.array(self_[3])
            dis = np.linalg.norm(offset)
            vec.append(dis)
        return vec


    def __game_state_to_features(self, game_state):
        # possition of the agent
        x, y = game_state['self'][3]
        # bomb or not
        has_bom = game_state['self'][2] > 0
        w, h = game_state['field'].shape

        # extract field features
        offset = 1
        # field = game_state['field'].copy()
        # field = np.zeros((h, w), dtype=int)
        # field[max(0, y - offset):min(h, y + offset + 1),
            #   max(0, x - offset):min(w, x + offset + 1)] = game_state['field'][max(0, y - offset):min(h, y + offset + 1),
                                                                            #    max(0, x - offset):min(w, x + offset + 1)]
        field = game_state['field'].copy()

        # where will be explosions
        explosion_map = self.__tile_to_explosion(game_state['bombs'], game_state['field'])

        distance_to_coins = self.__distance_to_coins(game_state['self'], game_state['coins'])

        vector_to_others = self.__vector_to_others(game_state['self'], game_state['others'])

        features = {
            'field': field,
            'explosion_map': explosion_map,
            'pos': (x, y),
            'has_bom': has_bom,
            'distance_to_coins': distance_to_coins,
            'vector_to_others': vector_to_others
        }

        return features

        
    def take_game_key(self, game_state):
        features = self.__game_state_to_features(game_state)
        hashing_key = self.__hashing_game_state(features)
        if hashing_key not in self.q_table:
            self.q_table[hashing_key] = {action: random.uniform(-2, 0) for action in self.actions}
        return hashing_key


    def __hashing_game_state(self, game_state):
        # Create a unique key for the game state
        field_key = tuple(game_state['field'].flatten())
        explosion_map_key = tuple(sorted(game_state['explosion_map']))
        pos_key = tuple(game_state['pos'])
        has_bom_key = game_state['has_bom']
        distance_to_coins_key = tuple(game_state['distance_to_coins'])
        vector_to_others_key = tuple(tuple(v) for v in game_state['vector_to_others'])

        return (field_key, explosion_map_key, pos_key, has_bom_key, distance_to_coins_key, vector_to_others_key) 



    def save_model(self):
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
