from agent_code.new_bie.Q_learn import QLearning

def setup(self):
    if self.train:
        self.model = QLearning()
        
    
def act(self, game_state: dict):
    return self.model.act(game_state)