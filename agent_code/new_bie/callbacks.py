from agent_code.new_bie.Q_learn import QLearning

def setup(self):
    self.model = QLearning(train=self.train, model_path="agent_code/new_bie/model/q_learn.pkl", reward_log="agent_code/new_bie/logs/reward_log.csv")
        
    
def act(self, game_state: dict):
    return self.model.act(game_state)