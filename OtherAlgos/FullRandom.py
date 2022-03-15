import numpy as np
import random
from Environment.GPEnvironments import GPMultiAgent
import matplotlib.pyplot as plt
from utils import plot_trajectory

seed = 232

np.random.seed(seed)
random.seed(seed)

# train
nav = np.genfromtxt('../Environment/example_map.csv', delimiter=',')
n_agents = 1
init_pos = np.array([[26, 21]])
initial_meas_locs = np.vstack((init_pos + [0,1],
                               init_pos + [0,-1],
                               init_pos + [1,0],
                               init_pos + [-1,0]))


env = GPMultiAgent(navigation_map=nav,
                   movement_length=3,
                   number_of_agents=n_agents,
                   initial_positions=init_pos,
                   initial_meas_locs=initial_meas_locs,
                   distance_budget=200,
                   device='cpu')


env.reset()
# N executions of the algorithm
N = 1
draw = True

class FullRandomAgent:

	def __init__(self, action_space, valid_action_function):
		self.action_space = action_space
		self.valid_action_function = valid_action_function

	def safe_select_action(self):

		valid = False
		while not valid:
			a = self.action_space.sample()
			valid = self.valid_action_function(a)

		return a

agent = FullRandomAgent(env.action_space, env.valid_action)

for t in range(N):

	env.reset()
	env.render()
	done = False

	while not done:

		a = agent.safe_select_action()

		s, r, done, i = env.step(a)

		print('MSE ', env.mse)
		env.render()

	plot_trajectory(env.axs[1], env.fleet.vehicles[0].waypoints[:, 1], env.fleet.vehicles[0].waypoints[:, 0],
	                z=None,
	                colormap='jet',
	                num_of_points=500, linewidth=2, k=3, plot_waypoints=True, markersize=0.5, zorder=4)
	plt.show(block=True)



