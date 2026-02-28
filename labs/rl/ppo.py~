import gymnasium as gym
import numpy as np
import rustoracerpy
from stable_baselines3 import PPO

env = gym.make("Rustoracer-v0", yaml="maps/berlin.yaml")
# train model
env.close()

env = gym.make("Rustoracer-v0", yaml="maps/berlin.yaml", render_mode="human")
# load model

obs, info = env.reset()
while True:
    action = np.array([0.0, 0.0])  # calculation action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()
