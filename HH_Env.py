"""
This file creates the Haunted House enviroment
"""

import gymnasium as gym 
import ale_py
import numpy as np

from collections import defaultdict

from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers import ResizeObservation
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import TransformObservation
from gymnasium.wrappers import TimeLimit 

gym.register_envs(ale_py)

# Make env grayscale, 1D, and smaller 
def process_env():
    env = gym.make("ALE/HauntedHouse-v5", render_mode=None) #render None for speed, "human" to see how it works
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=(16, 16))  
    env = TransformObservation(
        env,
        lambda obs: (obs // 32).astype(np.uint8),  
        gym.spaces.Box(0, 7, shape=(8, 8), dtype=np.uint8)
    )
    env = FlattenObservation(env)  # make observation 1D for tuple key
    
    env = TimeLimit(env,max_episode_steps= 5000)
    return env

