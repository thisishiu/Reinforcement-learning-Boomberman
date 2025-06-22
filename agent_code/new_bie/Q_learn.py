import events as e
import settings as s
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

COLS = s.COLS
ROWS = s.ROWS
TILE = {
    'WALL': -1,
    'FREE': 0,
    'CRATE': 1,
    'COIN': 2,
    'OTHER': 3,
    'DANGER': 4
}
ACTION = {
    'UP': 0,
    'RIGHT': 1,
    'DOWN': 2,
    'LEFT': 3,
    'WAIT': 4,
    'BOMB': 5
}


class QLearning:
    def __init__(self, train=True, epsilon=0.1, alpha=0.7, gamma=0.8, model_path=None, reward_log=None):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.train = train

        # self.q_table = np.random.rand(*(len(TILE)**5 * (COLS-2) * (ROWS-2) * 2, len(ACTION))).astype(np.float16)

        self.q_table = np.random.rand(*(len(TILE)**9 * 2, len(ACTION))).astype(np.float16)
        # [up, right, down, left, center, vec2others_x, vec2others_y, can_bomb]

        self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

        self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

        self.model_path = model_path
        if not train:
            self.load_model(model_path) 

        self.reward_log = reward_log
        if reward_log is not None:
            self.reward_log_list = []
            self.current_round_reward = 0


    def act(self, game_state):
        """
        Choose an action based on the current game state.
        Uses epsilon-greedy strategy for exploration vs exploitation.
        """
            
        state_features = self.take_game_features(game_state)
        state_key = self.feature_to_key(state_features)
        self.__bombed = None

        if not self.train or random.random() > self.epsilon:
            self.set_action(game_state)

            actions = {ac: self.q_table[state_key, ACTION[ac]] for ac in self.actions}
            # print(f"[!] Action: {actions}")
            action = max(actions, key=actions.get)

        else:
            # state_key = self.take_game_features(game_state)
            # self.set_action(state_features)
            action = random.choice(list(ACTION.keys()))

        # if action == 'BOMB':
        #     self.__bombed = [True, game_state['self'][3]]
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
        if e.BOMB_EXPLODED in events:
            self.__bombed = None
        if e.BOMB_DROPPED in events:
            self.__bombed = [True, new_game_state['self'][3]]
        if self.train:
            self.update_q_table(old_game_state, self_action, new_game_state, events)
        print("--------------------------------------")


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
        old_state_features = self.take_game_features(old_state)
        new_state_features = self.take_game_features(new_state)
        old_state_key = self.feature_to_key(old_state_features)
        new_state_key = self.feature_to_key(new_state_features)

        events = self.check_move(old_state, action, new_state, events)
        reward = self.reward_from_events(events)

        best_future_q = np.max(self.q_table[new_state_key])

        current_q = self.q_table[old_state_key, ACTION[action]]
        new_q = current_q * (1 - self.alpha) + self.alpha * (reward + self.gamma * best_future_q)

        print(f"[!] Real action: {action}")
        print(f"[!] Events: {events}")
        print(f"[!] Reward: {reward}")
        print(f"[!] Old Q-value: {current_q}, New Q-value: {new_q}, Best future Q: {best_future_q}")

        self.q_table[old_state_key, ACTION[action]] = new_q



    def reward_from_events(self, events):
        game_rewards = {
            e.INVALID_ACTION: -100, #
            e.RUN_IN_LOOP:-50,      #
            e.WAITED: -3,          
            e.GET_TRAPPED: -50,

            e.GOT_KILLED: -150,
            e.KILLED_SELF: -300,
            e.SURVIVED_ROUND: 3,

            e.MOVED_UP:-1,
            e.MOVED_RIGHT: -1,
            e.MOVED_DOWN: -1,
            e.MOVED_LEFT: -1,

            e.COIN_COLLECTED: 50,
            e.COIN_DISTANCE_REDUCED: 10,
            e.COIN_DISTANCE_INCREASED: -5,
            e.COIN_FOUND: 15, 
            
            e.BOMB_DROPPED: -10,
            e.BOMB_DROPPED_NEXT_TO_CRATE: 100,
            e.BOMB_DROPPED_NEXT_TO_OPPONENT: 200,
            e.BOMB_AVOIDED : 20,
            e.BOMB_DISTANCE_INCREASED: 15,
            
            e.CRATE_DESTROYED: 20,
            e.CRATE_DISTANCE_REDUCED: 7,
            e.CRATE_DISTANCE_INCREASED: -2,
            e.CRATE_FOUND: 2, 
            e.CRATE_WITHOUT_DROPPING_BOMB: -15,

            e.KILLED_OPPONENT: 500,
            e.OPPONENT_DISTANCE_REDUCED: 1,
            e.OPPONENT_DISTANCE_INCREASED: -5,
            e.OPPONENT_WITHOUT_DROPPING_BOMB: -15,

            e.GET_IN_DANGER: -100,
            e.IN_DANGER: -5,
            e.NOT_SAFE_FROM_BOMB: -50,
            e.SAFE_FROM_BOMB: 100,
            e.WAITING_IN_DANGER_ZONE: -30,
            e.NOT_MEANINGFUL_BOMB_DROP: -20
        }

        return sum(game_rewards.get(event, 0) for event in set(events))


    def check_move(self, old_game_state, self_action, new_game_state, events):
        subevent = []
        old_game_features = self.take_game_features(old_game_state)
        new_game_features = self.take_game_features(new_game_state)

        print(f"[!] Old game features: {old_game_features}")
        print(f"[!] New game features: {new_game_features}")

        # check if action move to bomb
        if self.__bombed is None and old_game_features[8] == TILE['DANGER'] and new_game_features[8] == TILE['DANGER']:
            subevent.append(e.WAITING_IN_DANGER_ZONE)

        if self_action != 'BOMB' and old_game_features[8] != TILE['DANGER'] and new_game_features[8] == TILE['DANGER']:
            subevent.append(e.GET_IN_DANGER)

        # check if agent move out of danger zone
        if old_game_features[8] == TILE['DANGER'] and new_game_features[8] != TILE['DANGER']:
            subevent.append(e.SAFE_FROM_BOMB)

        if self.__bombed is not None and old_game_features[8] == TILE['DANGER'] and new_game_features[8] != TILE['DANGER']:
            subevent.append(e.BOMB_AVOIDED)
            self.__bombed = None
        elif self.__bombed is not None and new_game_state['self'][3] == self.__bombed[1]:
            subevent.append(e.IN_DANGER)
    

        # check if meaningful bomb drop
        if self_action == 'BOMB':
            if TILE['CRATE'] in old_game_features[:8]:
                subevent.append(e.BOMB_DROPPED_NEXT_TO_CRATE)

        elif self_action == 'BOMB' and TILE['OTHER'] in old_game_features[:8]:
            subevent.append(e.BOMB_DROPPED_NEXT_TO_OPPONENT)

        elif self_action == 'BOMB' and TILE['CRATE'] not in old_game_features[:8] and TILE['OTHER'] not in old_game_features[:8]:
            subevent.append(e.NOT_MEANINGFUL_BOMB_DROP)

        # check if move close to other
        # if self.__l2_distance(new_game_features[5:7]) < self.__l2_distance(old_game_features[5:7]):
        #     subevent.append(e.OPPONENT_DISTANCE_REDUCED)

        return events + subevent
        

    def set_action(self, game_state):
        game_features = self.take_game_features(game_state)

        action = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

        # do not wait in danger zone
        if game_features[8] == TILE['DANGER']:
            action.remove('WAIT')

        # check tile can move
        if game_features[0] in [TILE['WALL'], TILE['CRATE']]:
            action.remove('UP')
        if game_features[2] in [TILE['WALL'], TILE['CRATE']]:
            action.remove('RIGHT')
        if game_features[4] in [TILE['WALL'], TILE['CRATE']]:
            action.remove('DOWN')
        if game_features[6] in [TILE['WALL'], TILE['CRATE']]:
            action.remove('LEFT')

        # check if can bomb
        if not self.__able_to_bomb(game_state):
            action.remove('BOMB')
        

        self.actions = action
        print(f"[!] Available actions: {self.actions}")
        return action




    def __able_to_bomb(self, game_state):
        game_key = self.take_game_features(game_state)
        # print(f"[!] Game key: {game_key}")
        if game_key[9] != 1:
            return False
        
        w, h = game_state['field'].shape
        x, y = game_state['self'][3]
        # map2check = game_state['field'][max(0, y-3):min(h, y+4), max(0, x-3):min(w, x+4)]
        map2check = game_state['field'].copy()

        if any(map2check[y-1, i] == TILE['FREE'] for i in range(max(0, x-3), x)) or any(map2check[y-1, i] == TILE['FREE'] for i in range(x+1, min(w, x+4))) or any(map2check[i, x-1] == TILE['FREE'] for i in range(max(0, y-3), y)) or any(map2check[i, x-1] == TILE['FREE'] for i in range(y+1, min(h, y+4))) or any(map2check[y+1, i] == TILE['FREE'] for i in range(max(0, x-3), x)) or any(map2check[y+1, i] == TILE['FREE'] for i in range(x+1, min(w, x+4))) or any(map2check[i, x+1] == TILE['FREE'] for i in range(max(0, y-3), y)) or any(map2check[i, x+1] == TILE['FREE'] for i in range(y+1, min(h, y+4))) or any(game_key[:9]) == TILE['OTHER']:
            return True
        return False

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
            power = 3
            for dx in range(1, power + 1):
                if x + dx >= w or field[y, x + dx] == TILE['WALL']:
                    break
                explosion_map.append((x + dx, y))  # Right
            for dx in range(1, power + 1):
                if x - dx < 0 or field[y, x - dx] == TILE['WALL']:
                    break
                explosion_map.append((x - dx, y))
            for dy in range(1, power + 1):
                if y + dy >= h or field[y + dy, x] == TILE['WALL']:
                    break
                explosion_map.append((x, y + dy))
            for dy in range(1, power + 1):
                if y - dy < 0 or field[y - dy, x] == TILE['WALL']:
                    break
                explosion_map.append((x, y - dy))
            explosion_map.append((x, y))  # Add the bomb's position itself

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
        game_state_field = game_state['field'].copy()
        explosion_map = self.__tile_to_explosion(game_state['bombs'], game_state_field)
        for ex in explosion_map:
            if 0 <= ex[0] < w and 0 <= ex[1] < h:
                # print(f"[!] Explosion at: {ex[0]}, {ex[1]}")
                game_state_field[ex[1], ex[0]] = TILE['DANGER']
        # print(f"[!] Game state field: \n{game_state_field.T}")
    
        for other in game_state['others']:
            game_state_field[other[3][1], other[3][0]] = TILE['OTHER']

        field = []
        field.append(game_state_field[y-1, x] if y > 0 else WALL)
        field.append(game_state_field[y-2, x] if y > 1 else WALL)
        field.append(game_state_field[y, x+1] if x < w-1 else WALL)
        field.append(game_state_field[y, x+2] if x < w-2 else WALL)
        field.append(game_state_field[y+1, x] if y < h-1 else WALL)
        field.append(game_state_field[y+2, x] if y < h-2 else WALL)
        field.append(game_state_field[y, x-1] if x > 0 else WALL)
        field.append(game_state_field[y, x-2] if x > 1 else WALL)
        field.append(game_state_field[y, x])  # Center tile

        vec2others = self.__vector_to_others(game_state['self'], game_state['others'])
        min_vec = min(vec2others, key=lambda v: np.linalg.norm(v)) if vec2others else [0, 0]

        can_bomb = 1 if has_bom and game_state['self'][2] > 0 else 0

        # return list(field) + list(min_vec) + [can_bomb]
        return list(field) + [can_bomb]


    def feature_to_key(self, features):
        base_sizes = [len(ACTION)]*9 + [2]
        index = 0
        for val, base in zip(features, base_sizes):
            index = index * base + val
        return index


    def take_game_features(self, game_state):
        features = self.__game_state_to_features(game_state)
        # key = self.key_to_index(features)
        return features


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
        except FileNotFoundError as e:
            print(f"[!] Q-table file not found: {e}. Starting fresh.")
            self.q_table = {}
