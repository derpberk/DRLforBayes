import numpy as np
import random
from Environment.GPEnvironments import GPMultiAgent
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
# N executions of the algorithm
N = 100

draw = True

def reverse_action(a):
	return (a + 6) % 12


for t in range(N):

	env.reset()

	if random.random() > 0.7:
		if env.fleet.vehicles[0].position[0] > env.navigation_map.shape[0] / 2:
			direction_UPDOWN = 3
		else:
			direction_UPDOWN = 7
		direction_LEFTRIGHT = 1
	else:
		if env.fleet.vehicles[0].position[0] > env.navigation_map.shape[0] / 2:
			direction_UPDOWN = 4
		else:
			direction_UPDOWN = 0
		direction_LEFTRIGHT = 2


	done = False
	action = direction_LEFTRIGHT
	vertical_need_flag = False
	env.render()

	while not done:

		horizontal_valid = env.valid_action(direction_LEFTRIGHT)
		vertical_valid = env.valid_action(direction_UPDOWN)

		# Si estamos retrocediendo porque no podemos subir
		if vertical_valid and vertical_need_flag:
			action = direction_UPDOWN
			vertical_need_flag = False
		else:
			# Si no es valida lateralmente:
			if not horizontal_valid:
				# Comprobamos si es valido verticalmente
				if vertical_valid:
					# Si lo es, tomamos un paso en vertical y cambiamos la direccion
					action = direction_UPDOWN
					direction_LEFTRIGHT = reverse_action(direction_LEFTRIGHT)
				else:
					# Si no lo es, cambiamos la direccion
					direction_LEFTRIGHT = reverse_action(direction_LEFTRIGHT)
					action = direction_LEFTRIGHT
					vertical_need_flag = True
			else:
				action = direction_LEFTRIGHT

		s, r, done, i = env.step(action)
		print("MSE: ", env.mse)
		env.render()

	plot_trajectory(env.axs[1], env.fleet.vehicles[0].waypoints[:, 1], env.fleet.vehicles[0].waypoints[:, 0],
	                z=None,
	                colormap='jet',
	                num_of_points=500, linewidth=2, k=3, plot_waypoints=True, markersize=0.5, zorder=4)
	plt.show(block=True)

