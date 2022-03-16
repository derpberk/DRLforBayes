import numpy as np
import random
from skopt.acquisition import gaussian_ei
from Environment.GPEnvironments import GPMultiAgent
import matplotlib.pyplot as plt
from utils import plot_trajectory
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor

seed = 45

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


env.seed(seed)
env.reset()
# N executions of the algorithm
N = 100


def reverse_action(a):
	return (a + 4) % 8

draw = True


gpr = GaussianProcessRegressor(kernel = C(1.0) * RBF(10.0))

for t in range(N):

	s = env.reset()
	done = False
	goal_achieved = True

	while not done:

		gpr.fit(env.measured_locations, env.measured_values)

		ei_map = np.zeros_like(env.navigation_map)
		ei_map[env.visitable_positions[:,0].astype(int), env.visitable_positions[:,1].astype(int)] = gaussian_ei(env.visitable_positions, gpr, xi=100.0)

		highest_std_location = np.unravel_index(np.argmax(ei_map), shape = env.navigation_map.shape)

		angle_dif = np.arctan2(highest_std_location[1]-env.fleet.vehicles[0].position[1], highest_std_location[0]-env.fleet.vehicles[0].position[0])
		angle_dif = 2*np.pi + angle_dif if angle_dif < 0 else angle_dif

		a = np.argmin(np.abs(angle_dif-env.fleet.vehicles[0].angle_set))

		while not env.valid_action(a):
			a = env.action_space.sample()

		s, r, done, i = env.step(a)

		env.render()
		print('MSE: ', env.mse)

	plt.show(block=bool)


	fig, ax = plt.subplots(1,1)

	true_map = np.zeros_like(env.navigation_map) * np.nan
	true_map[env.visitable_positions[:,0].astype(int), env.visitable_positions[:,1].astype(int)] = env.gt.GroundTruth_field
	error = 100*np.abs((env.mu - true_map) / (true_map + 1E-10))
	plt.imshow(error, vmin=0, vmax=100)

	plot_trajectory(ax, env.fleet.vehicles[0].waypoints[:, 1], env.fleet.vehicles[0].waypoints[:, 0],
	                z=None,
	                colormap='jet',
	                num_of_points=500, linewidth=2, k=3, plot_waypoints=True, markersize=0.5, zorder=4)

	print("Mean ERROR ", np.nanmean(error))
	plt.show(block=True)


