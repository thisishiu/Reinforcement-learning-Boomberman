import numpy as np
import pickle
import json
import random

class QLearning:
    def __init__(self, train:bool=True, epsilon:float=0.1, alpha:float=0.5, gamma:float=0.9):
        self.train = train
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
        self.q_table = {}   # Hash table to store Q-values
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
        
    def act(self, game_state):
        game_key = self.take_game_key(game_state)
        # Take all actions and their corresponding Q-values for the current game state
        actions_rewards = self.q_table.get(game_key, {})
        # if not actions_rewards:
        if not actions_rewards:
            # Initialize Q-values for all actions if not present
            actions_rewards = {action: random.uniform(-2,0) for action in self.actions}
            self.q_table[game_key] = actions_rewards
        best_action = max(actions_rewards, key=actions_rewards.get)
        if random.random() < self.epsilon:
            # Explore: choose a random action
            action = random.choice(self.actions)
        else:
            # Exploit: choose the best action based on Q-values
            action = best_action
            
        print(len(self.q_table.keys()))
        return action
    
    
        
    def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
        old_state_key = self.take_game_key(old_game_state)
        new_state_key = self.take_game_key(new_game_state)

    
    def take_game_key(self, game_state):
        """
        Process the game state and return a feature vector.
        This is necessary to use the game state in the Q-learning algorithm.
        """
        # Convert game state to a feature vector
        features = self.__game_state_to_features(game_state)
        # print(f"features: {features}")
        # Convert game state to a hashable format
        state_key = self.__hashing_game_state(features)
        
        return state_key
    
        
    def __hashing_game_state(self, game_state):
        """
        Convert the game state into a hashable format.
        This is necessary to use the game state as a key in the Q-table.
        """
        field = tuple(map(tuple, game_state["field"]))  # 2D ndarray → tuple of tuples
        explosion_map = tuple(map(tuple, game_state["explosion_map"]))  # same

        self_info = (
            game_state["self"][1],             # id
            game_state["self"][2],             # is_dead
            tuple(game_state["self"][3])       # position
        )

        bombs = tuple((tuple(pos), timer) for pos, timer in game_state["bombs"])
        coins = tuple(tuple(pos) for pos in game_state["coins"])
        others = [
            (
                game_state["others"][i][0],  # id
                game_state["others"][i][1],  # is_dead
            )
        ]
        

        return (
            # game_state["round"],
            # game_state["step"],
            field,
            explosion_map,
            self_info,
            bombs,
            coins
        )
        
        
    def __game_state_to_features(self, game_state):
        """
        Convert the game state into a feature vector.
        This is necessary to use the game state in the Q-learning algorithm.
        """
        
        # Extract relevant possitions and maps
        x_pos, y_pos = game_state['self'][3]
        h_grid, w_grid = game_state['field'].shape
        # Crop `offset pixel` of the field and explosion map around the agent's position 
        offset = 5
        game_state['field'] = game_state['field'][max(0, x_pos - offset):min(w_grid, x_pos + offset + 1), max(0, y_pos - offset):min(h_grid, y_pos + offset + 1)]
        game_state['explosion_map'] = game_state['explosion_map'][max(0, x_pos - offset):min(w_grid, x_pos + offset + 1), max(0, y_pos - offset):min(h_grid, y_pos + offset + 1)]

        return game_state
