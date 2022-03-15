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
N = 50
draw = True

def reverse_action(a):
    return (a + 4) % 8

dets = []
infos = []

env.is_eval = True

for t in range(N):

    env.reset()

    a = env.action_space.sample()

    new_a = a
    done = False

    while not done:

        valid = env.valid_action(a)
        while not valid:
            new_a = env.action_space.sample()
            while new_a == reverse_action(a):
                new_a = env.action_space.sample()
            valid = env.valid_action(new_a)
        a = new_a

        env.render()
        s, r, done, m = env.step(a)
        print("MSE: ", env.mse)

    plot_trajectory(env.axs[1], env.fleet.vehicles[0].waypoints[:, 1], env.fleet.vehicles[0].waypoints[:, 0],
                    z=None,
                    colormap='jet',
                    num_of_points=500, linewidth=2, k=3, plot_waypoints=True, markersize=0.5, zorder=4)
    plt.show(block=True)

