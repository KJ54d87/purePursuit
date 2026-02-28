import time
import math

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import rustoracerpy

WHEELBASE = 0.3302

env = gym.make("Rustoracer-v0", yaml="maps/berlin.yaml", render_mode="human")
obs, info = env.reset()
env_unwrapped: rustoracerpy.RustoracerEnv = env.unwrapped  # type: ignore
waypoints = env_unwrapped.skeleton(info["pose"])

look_ahead_distance = 5
speed_constant = 1.5
turning_constant = 1

#fig, axes = plt.subplots(1,1)
#axes.scatter(waypoints[:, 0], waypoints[:, 1])
#fig.show()

# look into future. If big turn in future: slow down! If no turn, keep going!

waypoints_shifted_one = np.roll(waypoints, 1)
direction_vectors = waypoints - waypoints_shifted_one
        
plt.savefig("plswork.png")
plt.close()

last_speed = 0

try:
    while True:
        loop_start = time.perf_counter()
        x, y, theta = info["pose"]
        pos = np.array([x, y])

        # Find lookahead point
        ahead_x = x + look_ahead_distance*math.cos(theta)
        ahead_y = y + look_ahead_distance*math.sin(theta)

        transformed_points = np.array(waypoints)
        transformed_points[:, 0] = (waypoints[:, 0]-x)*math.cos(-theta) - (waypoints[:, 1]-y)*math.sin(-theta)
        transformed_points[:, 1] = (waypoints[:, 0]-x)*math.sin(-theta) + (waypoints[:, 1]-y)*math.cos(-theta)

        distance = (waypoints[:, 0] - x)**2 +  (waypoints[:, 1] - y)**2

        best_score = 9999999
        best = 0
        
        for i in range(distance.shape[0]):
            if distance[i] - look_ahead_distance < 0 or transformed_points[i][0] < 0:
                continue
            if distance[i] - look_ahead_distance < best_score:
                best_score = distance[i] - look_ahead_distance
                best = i

        lookahead_point = waypoints[best]

        # Pure pursuit steering
        relative_x = lookahead_point[0] - x
        relative_y = lookahead_point[1] - y

        rel_ahead_x = (ahead_x - x)*math.cos(-theta) - (ahead_y-y)*math.sin(-theta)
        rel_ahead_y = (ahead_x - x)*math.sin(-theta) + (ahead_y-y)*math.cos(-theta)

        rel_look_x = transformed_points[best][0]
        rel_look_y = transformed_points[best][1]

        L = rel_look_x**2 + rel_look_y**2

        gamma = (2*(rel_look_y))/L
        

        steering_angle = turning_constant*gamma
        if steering_angle > .4189:
            steering_angle = .4189
        if steering_angle < -.4189:
            steering_angle = -.4189
        
        """
        valid_points = transformed_points[:, 0] >= 0
        valid_points = waypoints[valid_points]
        plt.plot(valid_points[:, 0], valid_points[:, 1])
        plt.scatter([lookahead_point[0]], [lookahead_point[1]], label = "lookeahead point")
        plt.scatter([x], [y], label = "car location")
        plt.scatter([ahead_x], [ahead_y], label = "ahead location")
        plt.scatter([x + look_ahead_distance*math.cos(theta+steering_angle)], [y + look_ahead_distance*math.sin(theta+steering_angle)], label = "drive direction")
        plt.legend()
        
        plt.savefig("plswork.png")
        plt.close()
        #plt.close()
        #"""
        """
        valid_points = transformed_points[:, 0] >= look_ahead_distance
        valid_points = waypoints[valid_points]
        valid_points2 = distance -look_ahead_distance > 0
        valid_points2 = waypoints[valid_points2]

        plt.scatter([rel_look_x], [rel_look_y], label = "lookeahead point")
        plt.plot(valid_points[:, 0], valid_points[:, 1])
        plt.plot(valid_points[:, 0], valid_points[:, 1])
        plt.scatter([rel_ahead_x], [rel_ahead_y], label = "ahead location")
        plt.scatter([look_ahead_distance*math.cos(theta + steering_angle)], [look_ahead_distance*math.sin(theta + steering_angle)], label = "drive direction")
        plt.scatter([0], [0], label = "car location")
        plt.legend()
        plt.savefig("rel.png")
        #plt.clf()
        plt.close()
        #"""
        

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
        next_point_x = x + look_ahead_distance*math.cos(theta + steering_angle)/2
        next_point_y = y + look_ahead_distance*math.sin(theta + steering_angle)/2
        
        distance = (waypoints[:, 0] - next_point_x)**2 +  (waypoints[:, 1] - next_point_y)**2

        transformed_points = np.array(waypoints)
        transformed_points[:, 0] = (waypoints[:, 0]-next_point_x)*math.cos(-theta) - (waypoints[:, 1]-next_point_y)*math.sin(-theta)
        transformed_points[:, 1] = (waypoints[:, 0]-next_point_x)*math.sin(-theta) + (waypoints[:, 1]-next_point_y)*math.cos(-theta)

        best_score = 99999999
        best = 0
        
        for i in range(distance.shape[0]):
            if distance[i] - look_ahead_distance/2 < 0 or transformed_points[i][0] < 0:
                continue
            if distance[i] - look_ahead_distance < best_score:
                best_score = distance[i] - look_ahead_distance/2
                best = i

        rel_look_x = transformed_points[best][0]
        rel_look_y = transformed_points[best][0]

        L = rel_look_x**2 + rel_look_y**2

        gamma = (2*(rel_look_y))/L

        future_steering = turning_constant * gamma

        """
        plt.plot(waypoints[:, 0], waypoints[:, 1])
        plt.scatter([lookahead_point[0]], [lookahead_point[1]], label = "lookeahead point")
        plt.scatter([x], [y], label = "car location")
        plt.scatter([x + look_ahead_distance*math.cos(theta)], [y + look_ahead_distance*math.sin(theta)], label = "car direction")
        plt.scatter([next_point_x], [next_point_y], label = "next location")
        plt.scatter([waypoints[best][0]], [waypoints[best][1]], label = "next next location")
        plt.scatter([x + look_ahead_distance*math.cos(theta+steering_angle)], [y + look_ahead_distance*math.sin(theta+steering_angle)], label = "drive direction")
        plt.legend()
        
        plt.savefig("plswork.png")
        plt.close()
        #"""
        
        speed = max(5, -.379 * (2.7)**((abs(steering_angle))*10) + 15)
        current_speed = (last_speed+speed)/2
        last_speed = speed
        print(current_speed, steering_angle)
        action = np.array(
            [steering_angle, 5]
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
