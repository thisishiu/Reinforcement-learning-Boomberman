from agent_code.new_bie.Q_learn import QLearning
import random

def setup_training(self):
    pass
    
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list[str]):
    return self.model.game_events_occurred(old_game_state, self_action, new_game_state, events)
    
def end_of_round(self, last_game_states, last_action, events):
    ...