from agent_code.new_bie.Q_learn import QLearning

model_path = "agent_code/new_bie/model/q_learn_11.pkl"
reward_log = "agent_code/new_bie/logs/reward_log_11_v2.csv"

def setup(self):
    self.model = QLearning(train=self.train, model_path=model_path, reward_log=reward_log)
        
def act(self, game_state: dict):
    return self.model.act(game_state)