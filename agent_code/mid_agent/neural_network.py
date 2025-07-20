import numpy as np
import settings as s
import pickle
import os
from datetime import datetime

class BasicNN:

    def __init__(self):
        pass

    def setup(self, input_size:int, output_size:int, gamma = 0.8, alpha = 2):

        self.input_size = input_size
        self.output_size = output_size

        self.theta = {
            "W": np.random.random(size=(output_size, input_size)),
            "B": np.random.random(size=(output_size, 1))
            }
        
        self.gamma = gamma
        self.alpha = alpha
        print(f"[NN] Init network randomly with in:{input_size}, out:{output_size}")

    def forward(self, state:np.ndarray) -> np.ndarray:
        if state.shape[0] != self.input_size:
            raise ValueError(f"Shape is not macthing, {state.shape} <> {self.input_size}")
        
        z = self.theta["W"] @ state + self.theta["B"]
        pi = self.softmax(z)
        return pi

    def update_theta(self):
        self.theta['W'] += self.alpha * self.grad_W
        self.theta['B'] += self.alpha * self.grad_B

    def setup_log_grad(self):
        self.log_grad = []

    def update_log_grad(self):
        self.grad_W = np.zeros_like(self.theta['W'])
        self.grad_B = np.zeros_like(self.theta['B'])

        for i in range(len(self.log_grad)):
            dw, db = self.log_likelihood(self.log_grad[i][0], self.log_grad[i][1], self.log_grad[i][2])
            self.grad_W += dw * self.G[i]
            self.grad_B += db * self.G[i]

    def setup_G(self):
        self.G = []

    # def update_G(self):
    #     if not hasattr(self, "G"):
    #         raise ValueError(f"G has not been intial yet!")
    #     G = self.G
    #     for i in range(len(G) - 1 , -1, -1):
    #         if i != len(G) - 1:
    #             G[i] = G[i+1] * self.gamma + G[i]

    #     self.G = (G - np.mean(G)) / (np.std(G) + 1e-8)
        
    def update_G(self):
        if not hasattr(self, "G"):
            raise ValueError(f"G has not been initial yet!")
        G = self.G
        discounted_G = np.zeros_like(G)
        discounted_G[-1] = G[-1]
        for i in range(len(G)-2, -1, -1):
            discounted_G[i] = discounted_G[i+1] * self.gamma + G[i]
        discounted_G = (discounted_G - np.mean(discounted_G)) / (np.std(discounted_G) + 1e-8)
        self.G = discounted_G

    def log_likelihood(self, pi:np.ndarray, choice:int, feature:np.ndarray):

        grad_za = np.zeros_like(self.theta['W'])
        grad_za[choice] = feature.flatten()
        sum_grad_zi = np.zeros_like(self.theta['W'])
        for i in range(len(pi)):
            sum_grad_zi[i] += feature.flatten() * pi[i]
        grad_W = grad_za - sum_grad_zi

        grad_ba = np.zeros_like(self.theta['B'])
        grad_ba[choice] = 1
        sum_grad_bi = np.zeros_like(self.theta["B"])
        for i in range(len(pi)):
            sum_grad_bi += pi[i]
        grad_B = grad_ba - sum_grad_bi

        return grad_W, grad_B


    def softmax(self, z: np.ndarray):
        z = z - np.max(z, axis=0, keepdims=True)
        e_z = np.exp(z)
        return e_z / np.sum(e_z, axis=0, keepdims=True)
    
    def save(self, round, path:str = None):
        if path is None:
            timestamp = datetime.now().strftime("%m-%d_%H-%M")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, "models",f"{timestamp}_{self.input_size}_{round}.plk")
        
        with open(path, "wb") as f:
            pickle.dump({
                "W": self.theta["W"],
                "B": self.theta["B"],
                "input_size": self.input_size,
                "output_size": self.output_size,
                "gamma": self.gamma,
                "alpha": self.alpha
            }, f)
        print(f"[NN] Saved model to {path}")

    def load(self, path: str= None):
        if path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_path = os.path.join(current_dir, "models")
            
            files = [f for f in os.listdir(models_path) if os.path.isfile(os.path.join(models_path, f))]
            
            if files:
                files.sort(key=lambda f: os.path.getctime(os.path.join(models_path, f)), reverse=True)
                
                file = files[0]
                path = os.path.join(models_path, file)
            else:
                raise FileNotFoundError("No files found in models/")

        with open(path, "rb") as f:
            data = pickle.load(f)
            self.input_size = data["input_size"]
            self.output_size = data["output_size"]
            self.gamma = data["gamma"]
            self.alpha = data["alpha"]
            self.theta = {
                "W": data["W"],
                "B": data["B"]
            }
        print(f"[NN] Loaded model from {path}")

    
if __name__ == "__main__":
    A = BasicNN()
    A.setup(5, 3)
    A.log_likelihood(np.array([0.1, 0.1, 0.9], dtype=float), 2, np.array([1,2,3,4,5]))
