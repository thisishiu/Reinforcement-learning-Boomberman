# Bomberman

This repo contains the final submission to a university course focussed on machine learning. The goal of the project was to implement reinforcement learning in a given framework within 90 working hours. The framework was developed for Bomberman, which is a multi-player game as described in https://en.wikipedia.org/wiki/Bomberman.

The framework provides the game board and multiple data about the game state such as the position of the own bomberman, the position of the opponents, the position and time step of bombs and much more.

The key challenges of the project are the following:
1. meaningful feature engineering: develop informative and comprehensive features of the game state by using the information provided by the framework and process it into meaning features
2. train the model until convergence efficiently in small computers: therefore we split the state space into disjoint sets and trained the agent with a script on the disjoint sets individually by using the computers of both team members
3. Implement a reward function that encourages the agent to weigh the benefits between escaping a dangerous condition and following a path that leads to some points, either by collecting a coin or bombing an enemy.


The final report can be found under the directory "report". In the same directory you can also find the description of the framework and the tasks.
          
         
