import numpy as np
import pandas as pd
import events as e
import settings as s
import os
import glob
from agent_code.mid_agent.neural_network import BasicNN
from collections import deque
from datetime import datetime

TILE = {
    'WALL': -1,
    'FREE': 0,
    'CRATE': 1,
    'COIN': 2,
    'OTHER': 3,
    'DANGER': 4
}

class MidModel:
    def __init__(self, train: bool = True):
        if train:
            self.network = BasicNN()
        else:
            """Read the model from file"""
            self.network = BasicNN()
            self.network.load()

        self.action = ["UP", 
                       "RIGHT",
                       "DOWN",
                       "LEFT",
                       "WAIT",
                       "BOMB"]
        
        self.logging = []
        self.q = deque(maxlen=s.BOMB_TIMER)
        
    def act(self, game_state:dict):
        feature = self.state_to_feature(game_state=game_state).reshape(-1, 1)
        if not hasattr(self.network, "theta"):
            self.network.setup(input_size=len(feature), output_size=6)
        
        self.old_feature = feature
        self.pi=self.network.forward(feature)
        return np.random.choice(self.action, p=self.pi.flatten())

    def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list[str]):
        """
        Handle game events that occur during training.
        """
        events = self.get_events(old_game_state, self_action, new_game_state, events)
        r = self.get_reward(events=events)
        t = old_game_state['step']
        n = old_game_state['round']

        # print(f"{self_action} - {events} | {r}")

        self.logging.append([n, t, r])

        if not hasattr(self.network, 'G') or self.network.G is None:
            self.network.sutup_G()
        if not hasattr(self.network, "log_grad") or self.network.log_grad is None:
            self.network.setup_log_grad()
        
        self.q.append(self_action)
        self.network.G.append(r)
        self.network.log_grad.append([self.pi.copy(), self.action.index(self_action), self.old_feature.copy()])



    def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
        """
        Handle the end of a round during training.
        """
        r = self.get_reward(events=events)
        self.network.G[-1] += r

        # print(f"{last_action} - {events} | {r}")

        self.network.update_G()
        self.network.update_log_grad()

        self.network.update_theta()

        self.network.G = None
        self.network.log_grad = None
        # print(f"[INF] End of round!")


    def get_events(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list[str]):
        # info
        old_cor = old_game_state["self"][3]
        new_cor = new_game_state["self"][3]
        had_bomd = old_game_state["self"][2]
        extend_events = []
        old_game_state["explosion_map"] = self.__explosion_map(old_game_state["bombs"], old_game_state["field"])
        new_game_state["explosion_map"] = self.__explosion_map(new_game_state["bombs"], new_game_state["field"])

        # check if agent get into explosion map
        if old_game_state["explosion_map"][old_cor[0], old_cor[1]] == 0 and new_game_state["explosion_map"][new_cor[0], new_cor[1]] != 0 and self_action not in ['WAIT ', 'BOMB']:
            extend_events.append(e.GET_IN_DANGER)

        # check if egent wait in danger
        if old_game_state["explosion_map"][old_cor[0], old_cor[1]] > 0 and self_action == 'WAIT':
            extend_events.append(e.WAITING_IN_DANGER_ZONE)

        # in danger zone
        if old_game_state["explosion_map"][old_cor[0], old_cor[1]] != 0 and new_game_state["explosion_map"][new_cor[0], new_cor[1]] != 0:
            extend_events.append(e.IN_DANGER)
        
        # bomb near crate
        if self_action == 'BOMB' and had_bomd and any(old_game_state['field'][old_cor[1]+i, old_cor[0]+j] == TILE['CRATE'] for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
            extend_events.append(e.BOMB_DROPPED_NEXT_TO_CRATE)

        # repeat an action
        if len(self.q) == self.q.maxlen and len(set(a for a in self.q)) < 3:
            extend_events.append(e.RUN_IN_LOOP)

        events = events + extend_events
        return events
    
    def get_reward(self, events):
        mapping = {
            e.MOVED_LEFT : -1,
            e.MOVED_RIGHT : -1,
            e.MOVED_UP : -1,
            e.MOVED_DOWN : -1,
            e.WAITED : 0,
            e.INVALID_ACTION : -5,
            e.VALID_ACTION : 0,
            e.RUN_IN_LOOP : -50,
            e.MOVED_IN_CYCLE : 0,

            e.BOMB_DROPPED : -1,
            e.BOMB_DROPPED_NEXT_TO_CRATE : 5,
            e.BOMB_DROPPED_NEXT_TO_OPPONENT : 0,
            e.BOMB_EXPLODED : 0,
            e.BOMB_AVOIDED : 0,
            e.BOMB_DISTANCE_INCREASED : 0,
            e.BOMB_NEXT_TO_CRATE : 0,
            e.NOT_SAFE_FROM_BOMB : 0,
            e.MEANINGFUL_BOMB_DROP : 0,
            e.NOT_MEANINGFUL_BOMB_DROP : 0,
            e.SAFE_FROM_BOMB : 0,
            e.NOT_SAFE_FROM_BOMB : 0,

            e.CRATE_DESTROYED : 10,
            e.CRATE_WITHOUT_DROPPING_BOMB : 0,
            e.CRATE_FOUND : 0,
            e.CRATE_DISTANCE_REDUCED : 0,
            e.CRATE_DISTANCE_INCREASED : 0,
            e.CRATE_REACHED :0,

            e.COIN_FOUND : 5,
            e.COIN_COLLECTED : 10,

            e.KILLED_OPPONENT : 100,
            e.KILLED_SELF : -1000,

            e.GOT_KILLED : -50,
            e.SURVIVED_ROUND : 100,

            e.OPPONENT_ELIMINATED : 20,
            e.OPPONENT_DISTANCE_REDUCED : 0,
            e.OPPONENT_DISTANCE_INCREASED : 0,
            e.OPPONENT_WITHOUT_DROPPING_BOMB : 0,

            e.COIN_DISTANCE_REDUCED : 0,
            e.COIN_DISTANCE_INCREASED : 0,

            e.WAITING_IN_DANGER_ZONE : -20,
            e.GET_IN_DANGER : -20,
            e.IN_DANGER : -15,
            e.CERTAIN_DEATH : 0,
            e.GET_TRAPPED : 0
        }

        return sum(mapping.get(event, 0) for event in events)

    def state_to_feature(self, game_state:dict):

        field = game_state['field'].flatten().tolist()

        bombs = []
        for i in range(3):
            if i < len(game_state["bombs"]):
                bombs.extend([game_state['bombs'][i][0][0], game_state['bombs'][i][0][1], game_state['bombs'][i][1]])
            else:
                bombs.extend([-1, -1, -1])

        # explosion_map = game_state["explosion_map"].flatten().tolist()
        explosion_map = self.__explosion_map(game_state["bombs"], game_state["field"]).flatten().tolist()

        coins = [game_state["coins"][0][0], game_state["coins"][0][1]] if game_state["coins"] else [-1, 
        -1]

        self_ = [game_state["self"][1], game_state["self"][2], game_state["self"][3][0], game_state["self"][3][1]]

        others = []
        for i in range(3):
            if i < len(game_state["others"]):
                others.extend([game_state["others"][i][2], game_state["others"][i][3][0], game_state["others"][i][3][1]])
            else:
                others.extend([-1, -1, -1])

        feature = field + bombs + explosion_map + coins + self_ + others
        return np.array(feature, dtype=int)

    def __explosion_map(self, bombs:list, field:np.ndarray):
        exp_map = np.zeros_like(field)

        # print(bombs)
        for b in bombs:
            x = b[0][0]
            y = b[0][1]
            t = b[1]

            for dx in range(x + 1, min(s.BOMB_POWER + x + 1, s.COLS)):
                if field[dx, y] != -1:
                    exp_map[dx, y] = max(t, 0)
                else:
                    break

            for _dx in range(max(x - s.BOMB_POWER, 0), x):
                if field[_dx, y] != -1:
                    exp_map[_dx, y] = max(t, 0)
                else:
                    break

            for dy in range(y + 1, min(s.ROWS, y + s.BOMB_POWER + 1)):
                if field[x, dy] != -1:
                    exp_map[x, dy] = max(t, 0)
                else:
                    break
            
            for _dy in range(max(0, y - s.BOMB_POWER), y):
                if field[x, _dy] != -1:
                    exp_map[x, _dy] = max(t, 0)
                else:
                    break

            exp_map[x, y] = max(t, 0)
            
        # print(exp_map)
        return exp_map


    def end(self, last_game_state: dict, last_action: str, events: list[str]):
        """
        Handle the end of training.
        """
        rounds = last_game_state['round']
        timestamp = datetime.now().strftime("%m-%d_%H-%M")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models",f"{timestamp}_{len(self.old_feature)}_{rounds}.plk")
        log_path = os.path.join(current_dir, "logs",f"{timestamp}_{len(self.old_feature)}_{rounds}.csv")

        df = pd.DataFrame(self.logging, columns=['round', "step", "reward"])
        df.to_csv(log_path, index=False)

        self.network.save(last_game_state['round'], model_path)
        print(f"[TRAIN] end train")