import events as e
import numpy as np
import os
import pickle
from typing import List
from collections import deque

class MidAgentModel:
    def __init__(self, train=True, epsilon=0.1, alpha=0.5, gamma=0.9, model_path=None, reward_log=None):
        self.train = train
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
        # Define state space dimensions
        # Direction to closest coin (8 directions + no coins)
        self.coin_directions = 9
        # Is the agent in danger from a bomb (yes/no)
        self.danger_states = 2
        # Are there crates nearby (yes/no)
        self.crates_nearby = 2
        # Can place bomb (yes/no)
        self.can_bomb = 2
        
        # Define number of actions (UP, RIGHT, DOWN, LEFT, WAIT, BOMB)
        self.n_actions = 6
        
        # Initialize Q-table as a numpy array
        # Shape: (coin_directions, danger_states, crates_nearby, can_bomb, n_actions)
        self.q_table = np.zeros((
            self.coin_directions, 
            self.danger_states, 
            self.crates_nearby, 
            self.can_bomb, 
            self.n_actions
        ))
        
        if train:
            self.model_path = model_path if model_path else "q_table.npy"
        if reward_log:
            self.reward_log = reward_log if reward_log else "reward_log.csv"
        # Load model if path is provided
        if model_path and os.path.isfile(model_path):
            self.load_model(model_path)
        
        # Performance tracking
        self.reward_history = deque(maxlen=100)
        self.reward_per_round = []
        self.current_round_reward = 0
        
        # Action mapping
        self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
        self.action_to_idx = {action: idx for idx, action in enumerate(self.actions)}
        
        # Updated comprehensive rewards
        self.rewards = {
            e.INVALID_ACTION: -100,
            e.RUN_IN_LOOP: -50,
            e.WAITED: -10,
            e.GET_TRAPPED: -50,

            e.GOT_KILLED: -150,
            e.KILLED_SELF: -300,
            e.SURVIVED_ROUND: 3,

            e.MOVED_UP: -1,
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
            e.BOMB_AVOIDED: 30,
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
        
        # Track recent positions to detect loops
        self.position_history = deque(maxlen=10)
        
        # Logger reference (will be set by agent)
        self.logger = None
    
    def save_model(self):
        """Save Q-table to a file."""
        np.save(self.model_path, self.q_table)
        # print(f"[~] Model saved to {self.model_path}")
    
    def load_model(self, filename):
        """Load Q-table from a file."""
        try:
            self.q_table = np.load(filename)
            # print(f"[~] Model loaded from {filename}")
        except:
            ...
            # print(f"[!] Could not load model from {filename}")
    
    def get_action(self, state, explore=True):
        """Get action using epsilon-greedy policy."""
        if explore and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(self.actions)
        else:
            # Exploit: best action from Q-table
            state_idx = self.state_to_index(state)
            action_idx = np.argmax(self.q_table[state_idx])
            return self.actions[action_idx]
    
    def state_to_index(self, state):
        """Convert state features to Q-table indices."""
        coin_dir, danger, crates, can_bomb = state
        return (coin_dir, danger, crates, can_bomb)
    
    def state_to_features(self, game_state):
        """Extract features from game state."""
        if game_state is None:
            return None
        
        # Extract agent position
        player_pos = game_state['self'][3]
        
        # Coin direction (0-7 for 8 directions, 8 for no coins)
        coin_dir = 8  # Default: no coins
        if game_state['coins']:
            # Find closest coin
            coins = game_state['coins']
            closest_coin = min(coins, key=lambda c: np.sqrt((c[0] - player_pos[0])**2 + (c[1] - player_pos[1])**2))
            
            # Calculate direction to coin
            dx = closest_coin[0] - player_pos[0]
            dy = closest_coin[1] - player_pos[1]
            
            # Convert to one of 8 directions
            if dx > 0 and dy == 0:
                coin_dir = 0  # East
            elif dx > 0 and dy > 0:
                coin_dir = 1  # Northeast
            elif dx == 0 and dy > 0:
                coin_dir = 2  # North
            elif dx < 0 and dy > 0:
                coin_dir = 3  # Northwest
            elif dx < 0 and dy == 0:
                coin_dir = 4  # West
            elif dx < 0 and dy < 0:
                coin_dir = 5  # Southwest
            elif dx == 0 and dy < 0:
                coin_dir = 6  # South
            elif dx > 0 and dy < 0:
                coin_dir = 7  # Southeast
        
        # Check for danger (bombs)
        danger = 0  # Default: safe
        if game_state['bombs']:
            for bomb_pos, bomb_timer in game_state['bombs']:
                if self.is_in_danger(player_pos, bomb_pos):
                    danger = 1
                    break
        
        # Check for crates nearby
        crates_nearby = 0  # Default: no crates nearby
        if 'field' in game_state:
            field = game_state['field']
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                x, y = player_pos[0] + dx, player_pos[1] + dy
                if 0 <= x < field.shape[0] and 0 <= y < field.shape[1] and field[x, y] == 1:
                    crates_nearby = 1
                    break
        
        # Can place bomb
        can_place_bomb = 1 if game_state['self'][2] else 0
        
        return (coin_dir, danger, crates_nearby, can_place_bomb)
    
    def is_in_danger(self, player_pos, bomb_pos):
        """Check if player is in danger from a bomb."""
        # Check if in blast radius (Manhattan distance <= 3 and in same row or column)
        if ((bomb_pos[0] == player_pos[0] and abs(bomb_pos[1] - player_pos[1]) <= 3) or 
            (bomb_pos[1] == player_pos[1] and abs(bomb_pos[0] - player_pos[0]) <= 3)):
            return True
        return False
    
    def update_q_table(self, old_state, action, new_state, reward):
        """Update Q-table using Q-learning formula."""
        if old_state is None or new_state is None:
            return
        
        # Get indices
        old_state_idx = self.state_to_index(old_state)
        action_idx = self.action_to_idx[action]
        new_state_idx = self.state_to_index(new_state)
        
        # Current Q-value
        old_q_value = self.q_table[old_state_idx][action_idx]
        
        # Best next action's Q-value
        best_next_q = np.max(self.q_table[new_state_idx])
        
        # Q-learning update
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * best_next_q - old_q_value)
        
        # Update Q-table
        self.q_table[old_state_idx][action_idx] = new_q_value
    
    def add_custom_events(self, old_game_state, action, new_game_state, events):
        """Add custom events based on state transitions."""
        custom_events = events.copy()
        
        if old_game_state is None or new_game_state is None:
            return custom_events
        
        # Player positions
        old_pos = old_game_state['self'][3]
        new_pos = new_game_state['self'][3]
        
        # Update position history for loop detection
        self.position_history.append(new_pos)
        
        # Detect running in loops
        if len(self.position_history) == 10 and len(set(self.position_history)) <= 3:
            custom_events.append(e.RUN_IN_LOOP)
        
        # Check if player is trapped (surrounded by walls/bombs with no escape)
        if self.is_trapped(new_game_state):
            custom_events.append(e.GET_TRAPPED)
        
        # Coin distance changes
        if old_game_state['coins'] and new_game_state['coins']:
            # Get closest coin in old and new state
            old_coin_dists = [self.manhattan_distance(old_pos, c) for c in old_game_state['coins']]
            new_coin_dists = [self.manhattan_distance(new_pos, c) for c in new_game_state['coins']]
            
            min_old_dist = min(old_coin_dists)
            min_new_dist = min(new_coin_dists)
            
            if min_new_dist < min_old_dist:
                custom_events.append(e.COIN_DISTANCE_REDUCED)
            elif min_new_dist > min_old_dist:
                custom_events.append(e.COIN_DISTANCE_INCREASED)
            
            # Detect if new coin was found (was not visible before)
            old_coin_count = len(old_game_state['coins'])
            new_coin_count = len(new_game_state['coins'])
            if new_coin_count > old_coin_count:
                custom_events.append(e.COIN_FOUND)
        
        # Bomb related events
        if action == "BOMB":
            field = new_game_state['field']
            # Check if bomb dropped next to crate
            crate_nearby = False
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                x, y = new_pos[0] + dx, new_pos[1] + dy
                if 0 <= x < field.shape[0] and 0 <= y < field.shape[1] and field[x, y] == 1:
                    crate_nearby = True
                    custom_events.append(e.BOMB_DROPPED_NEXT_TO_CRATE)
                    break
            
            # Check if bomb dropped next to opponent
            if new_game_state['others']:
                for other in new_game_state['others']:
                    other_pos = other[3]
                    if self.manhattan_distance(new_pos, other_pos) <= 3:
                        custom_events.append(e.BOMB_DROPPED_NEXT_TO_OPPONENT)
                        break
        
        # Check if moved away from bomb
        if old_game_state['bombs'] and new_game_state['bombs']:
            old_bomb_dists = [self.manhattan_distance(old_pos, bomb_pos) for bomb_pos, _ in old_game_state['bombs']]
            new_bomb_dists = [self.manhattan_distance(new_pos, bomb_pos) for bomb_pos, _ in new_game_state['bombs']]
            
            if min(new_bomb_dists) > min(old_bomb_dists):
                custom_events.append(e.BOMB_DISTANCE_INCREASED)
                # Also check if agent avoided danger zone
                old_in_danger = any(self.is_in_danger(old_pos, bomb_pos) for bomb_pos, _ in old_game_state['bombs'])
                new_in_danger = any(self.is_in_danger(new_pos, bomb_pos) for bomb_pos, _ in new_game_state['bombs'])
                if old_in_danger and not new_in_danger:
                    custom_events.append(e.BOMB_AVOIDED)
        
        # Crate related events
        if 'field' in old_game_state and 'field' in new_game_state:
            old_field = old_game_state['field']
            new_field = new_game_state['field']
            
            # Get closest crate distances
            old_crate_positions = [(x, y) for x in range(old_field.shape[0]) for y in range(old_field.shape[1]) if old_field[x, y] == 1]
            new_crate_positions = [(x, y) for x in range(new_field.shape[0]) for y in range(new_field.shape[1]) if new_field[x, y] == 1]
            
            # New crate found
            if len(new_crate_positions) > len(old_crate_positions):
                custom_events.append(e.CRATE_FOUND)
            
            # Check crate distances if crates exist
            if old_crate_positions and new_crate_positions:
                old_crate_dists = [self.manhattan_distance(old_pos, c) for c in old_crate_positions]
                new_crate_dists = [self.manhattan_distance(new_pos, c) for c in new_crate_positions]
                
                min_old_crate_dist = min(old_crate_dists) if old_crate_dists else float('inf')
                min_new_crate_dist = min(new_crate_dists) if new_crate_dists else float('inf')
                
                if min_new_crate_dist < min_old_crate_dist:
                    custom_events.append(e.CRATE_DISTANCE_REDUCED)
                elif min_new_crate_dist > min_old_crate_dist:
                    custom_events.append(e.CRATE_DISTANCE_INCREASED)
                
                # Missed opportunity to bomb crate
                if min_new_crate_dist == 1 and action != "BOMB" and old_game_state['self'][2]:
                    custom_events.append(e.CRATE_WITHOUT_DROPPING_BOMB)
        
        # Opponent related events
        if old_game_state['others'] and new_game_state['others']:
            old_opponent_dists = [self.manhattan_distance(old_pos, other[3]) for other in old_game_state['others']]
            new_opponent_dists = [self.manhattan_distance(new_pos, other[3]) for other in new_game_state['others']]
            
            min_old_opp_dist = min(old_opponent_dists)
            min_new_opp_dist = min(new_opponent_dists)
            
            if min_new_opp_dist < min_old_opp_dist:
                custom_events.append(e.OPPONENT_DISTANCE_REDUCED)
            elif min_new_opp_dist > min_old_opp_dist:
                custom_events.append(e.OPPONENT_DISTANCE_INCREASED)
            
            # Missed opportunity to bomb opponent
            if min_new_opp_dist <= 3 and action != "BOMB" and old_game_state['self'][2]:
                custom_events.append(e.OPPONENT_WITHOUT_DROPPING_BOMB)
        
        return custom_events
    
    def is_trapped(self, game_state):
        """Check if agent is trapped with no escape route."""
        player_pos = game_state['self'][3]
        field = game_state['field']
        bombs = game_state['bombs']
        
        # Check all four directions
        escape_routes = 0
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            x, y = player_pos[0] + dx, player_pos[1] + dy
            # Check if position is valid and not a wall or crate
            if 0 <= x < field.shape[0] and 0 <= y < field.shape[1] and field[x, y] == 0:
                # Check if this position is safe from bombs
                safe = True
                for bomb_pos, _ in bombs:
                    if self.is_in_danger((x, y), bomb_pos):
                        safe = False
                        break
                if safe:
                    escape_routes += 1
        
        # Trapped if no escape routes
        return escape_routes == 0
    
    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def calculate_reward(self, events):
        """Calculate total reward from events."""
        return sum(self.rewards.get(event, 0) for event in events)
        
    def setup_training(self):
        """Initialize Q-learning agent for training."""
        if self.logger:
            self.logger.info("Q-learning agent initialized for training")
        else:
            print("Q-learning agent initialized for training")
        
        # Reset position history
        self.position_history = deque(maxlen=10)

    def log(self):
        if self.reward_log is not None:
            with open(self.reward_log, 'w') as f:
                f.write("episode,reward\n")
                for i, r in enumerate(self.reward_per_round):
                    f.write(f"{i},{r}\n")      

    def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
        """Process game events for Q-learning."""
        if old_game_state is None or new_game_state is None:
            return
        
        # Add custom events
        events = self.add_custom_events(old_game_state, self_action, new_game_state, events)
        
        # Calculate reward
        reward = self.calculate_reward(events)
        self.current_round_reward += reward
        
        # Extract features from game states
        old_state = self.state_to_features(old_game_state)
        new_state = self.state_to_features(new_game_state)
        
        # Update Q-table
        self.update_q_table(old_state, self_action, new_state, reward)
        
        # Decay exploration rate if needed
        if self.epsilon > 0.1:
            self.epsilon *= 0.9995

    def end_of_round(self, last_game_state, last_action, events):
        """End of round processing for Q-learning."""
        # Add final rewards
        if e.KILLED_SELF in events or e.GOT_KILLED in events:
            # These rewards should already be in events, but double-check
            pass
        else:
            events.append(e.SURVIVED_ROUND)
        
        # Calculate final reward
        reward = self.calculate_reward(events)
        self.current_round_reward += reward
        
        # Record round reward
        self.reward_history.append(self.current_round_reward)
        self.reward_per_round.append(self.current_round_reward)
        self.current_round_reward = 0
        
        # Reset position history
        self.position_history.clear()
        
        # Save model periodically
        if len(self.reward_history) % 10 == 0:
            self.save_model()
            if self.logger:
                self.logger.info(f"Average reward over last 100 rounds: {np.mean(self.reward_history):.2f}")
                self.logger.info(f"Current exploration rate: {self.epsilon:.4f}")
            else:
                # print(f"Average reward over last 100 rounds: {np.mean(self.reward_history):.2f}")
                # print(f"Current exploration rate: {self.epsilon:.4f}")
                ...

    def end(self, last_game_state, last_action, events):
        """Final cleanup at end of training."""
        # Save final model
        self.save_model()
        
        if self.reward_log is not None:
            self.log()
        
        if self.logger:
            self.logger.info("Training completed - final Q-table saved")
            self.logger.info(f"Final exploration rate: {self.epsilon:.4f}")
            if len(self.reward_per_round) > 0:
                self.logger.info(f"Average reward over all rounds: {np.mean(self.reward_per_round):.2f}")
        else:
            print("Training completed - final Q-table saved")
            print(f"Final exploration rate: {self.epsilon:.4f}")
            if len(self.reward_per_round) > 0:
                print(f"Average reward over all rounds: {np.mean(self.reward_per_round):.2f}")


