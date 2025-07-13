from agent_code.mid_agent.Mid_model import MidModel

def setup(self):
    self.nn = MidModel(train=self.train)
        
def act(self, game_state: dict):
    return self.nn.act(game_state)