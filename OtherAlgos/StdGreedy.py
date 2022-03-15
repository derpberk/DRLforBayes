import numpy as np
import random
from Environment.GPEnvironments import GPMultiAgent
from Environment.EnvironmentUtils import AStarPlanner
import matplotlib.pyplot as plt
from utils import plot_trajectory

seed = 232

np.random.seed(seed)
random.seed(seed)

# train
nav = np.genfromtxt('../Environment/ypacarai_map.csv')
n_agents = 1
init_pos = np.array([[26, 21]])*2
initial_meas_locs = np.vstack((init_pos + [0,5],
                               init_pos + [0,-5],
                               init_pos + [5,0],
                               init_pos + [-5,0]))


env = GPMultiAgent(navigation_map=nav,
                   movement_length=10,
                   number_of_agents=n_agents,
                   initial_positions=init_pos,
                   initial_meas_locs=initial_meas_locs,
                   distance_budget=300,
                   number_of_actions=12,
                   device='cpu')


env.reset()
planner = AStarPlanner(env.navigation_map, 1, 0.5)
# N executions of the algorithm
N = 100


def reverse_action(a):
	return (a + 4) % 8

draw = True

for t in range(N):

	s = env.reset()
	done = False
	goal_achieved = True

	while not done:

		highest_std_location = np.array(np.unravel_index(np.argmax(s[-1]), shape=s[-1].shape))
		path = np.asarray(planner.planning(list(highest_std_location), list(env.fleet.vehicles[0].position))).T

		distance_mask = np.where(np.linalg.norm(path-env.fleet.vehicles[0].position, axis=1) < 4)[0][-1]
		next_position = path[distance_mask]

		angle_dif = np.arctan2(next_position[1]-env.fleet.vehicles[0].position[1], next_position[0]-env.fleet.vehicles[0].position[0])
		angle_dif = 2*np.pi + angle_dif if angle_dif < 0 else angle_dif

		a = np.argmin(np.abs(angle_dif-env.fleet.vehicles[0].angle_set))

		while not env.valid_action(a):
			a = env.action_space.sample()

		s, r, done, i = env.step(a)

		env.render()


	plot_trajectory(env.axs[1], env.fleet.vehicles[0].waypoints[:, 1], env.fleet.vehicles[0].waypoints[:, 0],
	                z=None,
	                colormap='jet',
	                num_of_points=500, linewidth=2, k=3, plot_waypoints=True, markersize=0.5, zorder=4)
	plt.show(block=True)

