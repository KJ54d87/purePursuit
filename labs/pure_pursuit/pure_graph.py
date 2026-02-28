import time
import math

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


import rustoracerpy

WHEELBASE = 0.3302

env = gym.make("Rustoracer-v0", yaml="maps/berlin.yaml", render_mode="human")
obs, info = env.reset()
env_unwrapped: rustoracerpy.RustoracerEnv = env.unwrapped  # type: ignore
waypoints = env_unwrapped.skeleton(info["pose"])

look_ahead_distance = .4
turning_constant = -.1

#fig, axes = plt.subplots(1,1)
#axes.scatter(waypoints[:, 0], waypoints[:, 1])
#fig.show()
lookahead_point = [0,0]
x = 0
y= 0
ahead_x = 0
ahead_y = 0
theta = 0
steering_angle = 0
fig, ax = plt.subplots()

plt.plot(waypoints[:, 0], waypoints[:, 1])
plt.scatter([lookahead_point[0]], [lookahead_point[1]], label = "lookeahead point")
plt.scatter([x], [y], label = "car location")
plt.scatter([ahead_x], [ahead_y], label = "ahead location")
plt.scatter([x + look_ahead_distance*math.cos(theta+steering_angle)], [look_ahead_distance*math.sin(theta+steering_angle)], label = "drive direction")
plt.legend()
plt.draw()

def step(frame):
    try:
        loop_start = time.perf_counter()
        x, y, theta = info["pose"]
        pos = np.array([x, y])

        # Find lookahead point
        ahead_x = x + look_ahead_distance*math.cos(theta)
        ahead_y = y + look_ahead_distance*math.sin(theta)

        distance = (waypoints[:, 0] - ahead_x)**2 +  (waypoints[:, 1] - ahead_y)**2
        loc = np.argmin(distance)

        lookahead_point = waypoints[loc]

        #theta = (3.1415)/2 - theta

        # Pure pursuit steering
        relative_x = lookahead_point[0] - x
        relative_y = lookahead_point[1] - y

        rel_ahead_x = (ahead_x - x)*math.cos(-theta) - (ahead_y-y)*math.sin(-theta)
        rel_ahead_y = (ahead_x - x)*math.sin(-theta) + (ahead_y-y)*math.cos(-theta)

        rel_look_x = (lookahead_point[0] - x)*math.cos(-theta) - (lookahead_point[1]-y)*math.sin(-theta)
        rel_look_y = (lookahead_point[0] - x)*math.sin(-theta) + (lookahead_point[1]-y)*math.cos(-theta)

        L = rel_look_x**2 + rel_look_y**2

        gamma = (2*abs(rel_look_y))/L

        steering_angle = turning_constant*gamma
        if steering_angle > .4189:
            steering_angle = .4189
        if steering_angle < -.4189:
            steering_angle = -.4189
        
        #plt.savefig("plswork.png")
        #plt.close()
        """

        plt.scatter([rel_look_x], [rel_look_x], label = "lookeahead point")
        plt.scatter([rel_ahead_x], [rel_ahead_y], label = "ahead location")
        plt.scatter([x + look_ahead_distance*math.cos(steering_angle)], [look_ahead_distance*math.sin(steering_angle)], label = "drive direction")
        plt.scatter([x], [y], label = "car location")
        plt.legend()
        plt.savefig("rel.png")
        plt.close()
        """
        """
        plt.scatter([rel_ahead_x], [rel_ahead_y], label = "front")
        #plt.scatter([rel_look_x], [rel_look_y], label = "lookahead location")
        plt.scatter([x], [y], label = "car location")
        plt.legend()
        plt.savefig("rel.png")
        plt.close()
        """
        
        #print()
        #print(theta, gamma, steering_angle)

        action = np.array(
            [steering_angle, 30.0]
        )  # calculation action (steer, speed). Steer: [full right=-0.4189, full left=0.4189], Speed: [0.0, 20.0] (m/s)
        obs, reward, terminated, truncated, info = env.step(action)
        #env.render()
        elapsed = time.perf_counter() - loop_start
        time.sleep(max(0.0, 1.0 / 100.0 - elapsed))
        if terminated or truncated:
            obs, info = env.reset()
    except KeyboardInterrupt:
        pass

env.close()

ani = animation.FuncAnimation(fig=fig, func=step, frames=40, interval=30)
plt.show()
