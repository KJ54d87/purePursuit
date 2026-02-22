import time

import gymnasium as gym
import numpy as np

import rustoracerpy

WHEELBASE = 0.3302

env = gym.make("Rustoracer-v0", yaml="maps/berlin.yaml", render_mode="human")
obs, info = env.reset()
env_unwrapped: rustoracerpy.RustoracerEnv = env.unwrapped  # type: ignore
waypoints = env_unwrapped.skeleton(info["pose"])

try:
    while True:
        loop_start = time.perf_counter()
        x, y, theta = info["pose"]
        pos = np.array([x, y])

        # Find lookahead point

        # Pure pursuit steering

        action = np.array(
            [-1, 20.0]
        )  # calculation action (steer, speed). Steer: [full right=-0.4189, full left=0.4189], Speed: [0.0, 20.0] (m/s)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        elapsed = time.perf_counter() - loop_start
        time.sleep(max(0.0, 1.0 / 100.0 - elapsed))
        if terminated or truncated:
            obs, info = env.reset()
except KeyboardInterrupt:
    pass
finally:
    env.close()
