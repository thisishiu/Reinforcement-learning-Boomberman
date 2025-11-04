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
        self.q = deque(maxlen=int(s.BOMB_TIMER * 2.5))
        
    def act(self, game_state:dict):
        feature = self.state_to_feature(game_state=game_state).reshape(-1, 1)
        if not hasattr(self.network, "theta"):
            self.network.setup(input_size=len(feature), output_size=6)
        
        self.old_feature = feature
        self.pi=self.network.forward(feature)

        # print(np.array2string(self.pi.T, separator=', ', suppress_small=True, max_line_width=np.inf, formatter={'float_kind': lambda x: f"{x:.8f}"}))
        # return 'WAIT'
        # return np.random.choice(self.action)
        return np.random.choice(self.action, p=self.pi.flatten())


    def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list[str]):
        """
        Handle game events that occur during training.
        """
        events = self.get_events(old_game_state, self_action, new_game_state, events)
        r = self.get_reward(events=events)
        t = old_game_state['step']
        n = old_game_state['round']

        print(f"{self_action} - {events} | {r}")

        self.logging.append([n, t, r])
        self.q.append(self_action)

        if not hasattr(self.network, 'G') or self.network.G is None:
            self.network.setup_G()
        if not hasattr(self.network, "log_grad") or self.network.log_grad is None:
            self.network.setup_log_grad()
        
        self.network.G.append(r)
        self.network.log_grad.append([self.pi.copy(), self.action.index(self_action), self.old_feature.copy()])
        
        # self.network.update_G()
        # self.network.update_log_grad()
        # self.network.update_theta()
        
        # self.network.G = []
        # self.network.log_grad = []

    def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
        """
        Handle the end of a round during training.
        """
        r = self.get_reward(events=events)
        self.network.G[-1] += r
        # self.network.G.append(r)
        # self.network.log_grad.append([self.pi.copy(), self.action.index(last_action), self.old_feature.copy()])

        print(f"{last_action} - {events} | {r}")

        self.network.update_G()
        self.network.update_log_grad()
        self.network.update_theta()

        self.network.G = []
        self.network.log_grad = []
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

        # if agent avoid the bomb
        if old_game_state['explosion_map'][old_cor[0], old_cor[1]] != 0 and new_game_state["explosion_map"][new_cor[0], new_cor[1]] == 0 and old_cor != new_cor:
            extend_events.append(e.BOMB_AVOIDED)
        
        # bomb near crate
        if self_action == 'BOMB' and had_bomd and any(old_game_state['field'][old_cor[0]+i, old_cor[1]+j] == TILE['CRATE'] for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
            extend_events.append(e.BOMB_DROPPED_NEXT_TO_CRATE)

        # repeat an action
        if len(self.q) == self.q.maxlen and len(set(a for a in list(self.q)[s.BOMB_TIMER-1:self.q.maxlen])) < 3:
            extend_events.append(e.RUN_IN_LOOP)

        # if there are some ways to run after bomb:
        if self_action == "BOMB" and had_bomd and not self.__is_safe2bomb(old_cor, old_game_state['field']):
            extend_events.append(e.THERE_IS_NO_WAY_RUN)

        # distance to others
        vec2other = self.__vec2others(old_game_state)
        matrix2other = self.__matrix2others(vec2other)
        matching = ["UP", "RIGHT", "DOWN", "LEFT"]
        if self_action not in ["BOMB", "WAIT"] and 'INVALID_ACTION' not in events and matrix2other[matching.index(self_action)] == 1:
            extend_events.append(e.OPPONENT_DISTANCE_REDUCED)

        # distance to coin
        if old_game_state["coins"]:
            coins = old_game_state["coins"][0]
            vec2coin = [coins[0] - new_cor[0], coins[1] - new_cor[1]]
            matrix2coin = self.__matrix2others(vec2coin)
            if self_action not in ["BOMB", "WAIT"] and 'INVALID_ACTION' not in events and matrix2coin[matching.index(self_action)] == 1:
                extend_events.append(e.COIN_DISTANCE_REDUCED)
                
        # drop bomb next to opponent
        if self_action == 'BOMB' and had_bomd and old_game_state["others"]:
            for other in old_game_state["others"]:
                if other[3][0] in range(old_cor[0] - 1, old_cor[0] + 2) and other[3][1] in range(old_cor[1] - 1, old_cor[1] + 2):
                    extend_events.append(e.BOMB_DROPPED_NEXT_TO_OPPONENT)
                    break
                
        # if agent do not move for a long time
        if len(self.q) == self.q.maxlen and len(set(a for a in list(self.q)[s.BOMB_TIMER:self.q.maxlen])) == 1:
            extend_events.append(e.LONG_TIME_1_ACTION)
        
        # if agent no bomb for a long time
        if len(self.q) == self.q.maxlen and len(set(a for a in self.q if a == "BOMB")) == 0:
            extend_events.append(e.LONG_TIME_NO_BOMB)

        events = events + extend_events
        return events
    
    def __matrix2others(self, vec2other):
        a = [vec2other[1] < 0, vec2other[0] > 0, vec2other[1] > 0, vec2other[0] < 0]
        return [int(i) for i in a]

    def __vec2others(self, game_state: dict):
        x, y = game_state['self'][3][0], game_state['self'][3][1]

        if game_state["others"]:
            return [game_state["others"][0][3][0] - x, game_state["others"][0][3][1] - y]
        else:
            return [0, 0]

    def __is_safe2bomb(self, pos:tuple, field:np.ndarray):
        x = pos[0]
        y = pos[1]
        
        for i in range(x + 1, min(s.COLS, x + 5)):
            # print(f"{i},{y}: {field[i, y]}")
            if field[i, y] != 0:
                break
            # print(f"-> {i},{y+1}: {field[i, y + 1]}; {i},{y-1}: {field[i, y-1]}")
            if field[i, y + 1] == 0 or field[i, y - 1] == 0 or i == x+4:
                return True    
            
        for i in range(x - 1, max(-1, x - 5), -1):
            # print(f"{i},{y}: {field[i, y]}")  
            if field[i, y] != 0:
                break
            # print(f"-> {i},{y+1}: {field[i, y + 1]}; {i},{y-1}: {field[i, y-1]}")
            if field[i, y + 1] == 0 or field[i, y - 1] == 0 or i == x-4:
                return True  
        
        for i in range(y + 1, min(s.ROWS, y + 5)):
            # print(f"{x},{i}: {field[x, i]}")
            if field[x, i] != 0:
                break
            # print(f"-> {x+1},{y}: {field[x+1, y]}; {x-1},{y}: {field[x-1, y]}")
            if field[x + 1, i] == 0 or field[x - 1, i] == 0 or i == y+4:
                return True
        
        for i in range(y - 1, max(-1, y - 5), -1):
            # print(f"{x},{i}: {field[x, i]}")
            if field[x, i] != 0:
                break
            # print(f"-> {x+1},{y}: {field[x+1, y]}; {x-1},{y}: {field[x-1, y]}")
            if field[x + 1, i] == 0 or field[x - 1, i] == 0 or y == y-4:
                return True

        return False

    def get_reward(self, events):
        mapping = {
            e.MOVED_LEFT : 1,
            e.MOVED_RIGHT : 1,
            e.MOVED_UP : 1,
            e.MOVED_DOWN : 1,
            e.WAITED : 0,
            e.INVALID_ACTION : -5,
            e.VALID_ACTION : 0,
            e.RUN_IN_LOOP : -50,

            e.BOMB_DROPPED : -1,
            e.BOMB_DROPPED_NEXT_TO_CRATE : 5,
            e.BOMB_DROPPED_NEXT_TO_OPPONENT : 20,
            e.BOMB_EXPLODED : 0,
            e.BOMB_AVOIDED : 10,
            e.BOMB_DISTANCE_INCREASED : 0,
            e.NOT_SAFE_FROM_BOMB : 0,
            e.MEANINGFUL_BOMB_DROP : 0,
            e.NOT_MEANINGFUL_BOMB_DROP : 0,
            e.SAFE_FROM_BOMB : 0,

            e.CRATE_DESTROYED : 10,
            e.CRATE_WITHOUT_DROPPING_BOMB : 0,
            e.CRATE_FOUND : 0,
            e.CRATE_DISTANCE_REDUCED : 0,
            e.CRATE_DISTANCE_INCREASED : 0,
            e.CRATE_REACHED :0,

            e.COIN_FOUND : 5,
            e.COIN_COLLECTED : 12,

            e.KILLED_OPPONENT : 100,
            e.KILLED_SELF : -100,

            e.GOT_KILLED : -50,
            e.SURVIVED_ROUND : 100,

            e.OPPONENT_ELIMINATED : 20,
            e.OPPONENT_DISTANCE_REDUCED : 10,
            e.OPPONENT_DISTANCE_INCREASED : 0,
            e.OPPONENT_WITHOUT_DROPPING_BOMB : 0,

            e.COIN_DISTANCE_REDUCED : 10,
            e.COIN_DISTANCE_INCREASED : 0,

            e.LONG_TIME_NO_BOMB : -20,
            e.LONG_TIME_1_ACTION : -20,
            e.THERE_IS_NO_WAY_RUN : - 20,
            e.WAITING_IN_DANGER_ZONE : -20,
            e.GET_IN_DANGER : -20,
            e.IN_DANGER : -15,
            e.CERTAIN_DEATH : 0,
            e.GET_TRAPPED : 0
        }

        return sum(mapping.get(event, 0) for event in events)

    def state_to_feature(self, game_state:dict):

        x, y = game_state["self"][3][0], game_state["self"][3][1]

        field = self.__map_around((x, y), game_state['field'], -1).flatten()

        bombs = []
        for i in range(2):
            if i < len(game_state["bombs"]):
                bombs.extend([game_state['bombs'][i][0][0], game_state['bombs'][i][0][1]])
            else:
                bombs.extend([-1, -1])
        bomb = np.array(bombs)
        bomb = bomb / s.COLS # or s.ROWS        

        explosion_around = self.__map_around((x, y), self.__explosion_map(game_state["bombs"], game_state["field"])).flatten()
        explosion_around = explosion_around / s.BOMB_TIMER

        coins = [game_state["coins"][0][0], game_state["coins"][0][1]] if game_state["coins"] else [-1, -1]

        # current score, has bomb?, x, y
        has_bomb = [game_state["self"][2]]

        others = []
        #---------- 3 others -----------------------------
        # for i in range(3):
        #     if i < len(game_state["others"]):
        #         others.extend([game_state["others"][i][2], game_state["others"][i][3][0], game_state["others"][i][3][1]])
        #     else:
        #         others.extend([-1, -1, -1])
        #-------------------------------------------------

        #---------- 1 other ------------------------------
        if game_state["others"]:
            others.extend([game_state["others"][0][2], game_state["others"][0][3][0] / s.COLS, game_state["others"][0][3][1] / s.COLS])
        else:
            others.extend([-1, -1, -1])
        #-------------------------------------------------

        vec2coin = np.array([coins[0] - x, coins[1] - y]) / s.COLS
        vec2other = np.array([others[0] - x, others[1] - y]) / s.COLS
        
        # matrix2coin = self.__matrix2others(vec2coin)
        # matrix2other = self.__matrix2others(vec2other)

        feature = np.concatenate([[x/s.COLS], [y/s.COLS], field, has_bomb, explosion_around, vec2coin,  vec2other])

        # print(feature)
        return feature
    
    def __map_around(self, pos:list, explosion_map:np.ndarray, padding=0, offset=7):
        x = pos[0]
        y = pos[1]

        explosion_aronud = np.zeros(shape=(2*offset+1, 2*offset+1), dtype=int) + padding

        a = 0
        for i in range(x - offset, x + offset + 1):
            b = 0
            for j in range(y - offset, y + offset + 1):
                if 0 <= i < s.COLS and 0 <= j < s.ROWS:
                    explosion_aronud[a, b] = explosion_map[i, j]
                b += 1
            a += 1

        return explosion_aronud

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

            for _dx in range(x-1, max(x - s.BOMB_POWER - 1, -1), -1):
                if field[_dx, y] != -1:
                    exp_map[_dx, y] = max(t, 0)
                else:
                    break

            for dy in range(y + 1, min(s.ROWS, y + s.BOMB_POWER + 1)):
                if field[x, dy] != -1:
                    exp_map[x, dy] = max(t, 0)
                else:
                    break
            
            for _dy in range(y - 1, max(-1, y - s.BOMB_POWER -1), -1):
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