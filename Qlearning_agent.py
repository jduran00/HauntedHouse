import gymnasium as gym 
import ale_py
import numpy as np
from collections import defaultdict
from HH_Env import process_env

env = process_env()

#Create Haunted House Env
#env = gym.make("ALE/HauntedHouse-v5", render_mode="human")

# Structure influnced by gymnasium's training an agent documentation and berkleys Pacman project 

# Create agent
class HHAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
    
      self.env = env
      self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

      self.lr = learning_rate
      self.discount_factor = discount_factor

      self.epsilon = initial_epsilon
      self.epsilon_decay = epsilon_decay
      self.final_epsilon = final_epsilon

      self.training_error = []

    
    def getLegalActions(self, state):
        "Return a list of all possible actions"
        return list(range(self.env.action_space.n))
    
    def getQValue(self, state, action):
        "Return Q value"
        return self.q_values[state][action]
    
    def computeValueFromQValues(self, state):
        "Return max value from possible Q values"
        legal_actions = self.getLegalActions(state)
        if not legal_actions:
            return 0.0
        return np.max([self.getQValue(state, a) for a in legal_actions])

    def computeActionFromQValues(self, state):
        "Return action from possible Q values"
        legal_actions = self.getLegalActions(state)
        if not legal_actions:
            return None
        q_vals = np.array([self.getQValue(state, a) for a in self.getLegalActions(state)])
        max_q = np.max(q_vals)
        #best_actions = [a for a in legal_actions if q_vals[a] == max_q]
        #best_actions = [a for a in legal_actions if np.isclose(q_vals[a], max_q)]
        best_actions = [a for a, q in zip(legal_actions, q_vals) if np.isclose(q, max_q)]

        return np.random.choice(best_actions)
        
    

    def getAction(self, obs):
        """
        Return action to take in current state. 
        """
        state = tuple(obs.tolist())
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.computeActionFromQValues(state)
        
    def update(self, state, action, next_state, reward):
        """
        Q-Value update

        """
        next_q = self.computeValueFromQValues(next_state)
        curr_q = self.getQValue(state, action)
        target = reward + self.discount_factor * next_q

        # Update rule
        updatedQ = (1 - self.lr) * curr_q + self.lr * target
        self.q_values[state][action] = updatedQ
        self.training_error.append(abs(curr_q - target))

    def decayEpsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

#create agent
"""agent = HHAgent(
    env=env,
    learning_rate=0.1,
    initial_epsilon=1.0,
    epsilon_decay=0.995,
    final_epsilon=0.05,
    discount_factor=0.99
)"""
