def setup_training(self):
    print(f"[TRAIN] setup training")

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list[str]):
    """
    Handle game events that occur during training.
    """
    return self.nn.game_events_occurred(old_game_state, self_action, new_game_state, events)

def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    """
    Handle the end of a round during training.
    """
    return self.nn.end_of_round(last_game_state, last_action, events)

def end(self, last_game_state: dict, last_action: str, events: list[str]):
    """
    Handle the end of training.
    """
    return self.nn.end(last_game_state, last_action, events)
 