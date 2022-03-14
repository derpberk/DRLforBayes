from Environment.GPEnvironments import GPMultiAgent
import numpy as np
from DeepAgent.Agent.DuelingDQNAgent import DuelingDQNAgent

nav = np.genfromtxt('../Environment/example_map.csv', delimiter=',')
n_agents = 1
init_pos = np.array([[26, 21]])
initial_meas_locs = np.vstack((init_pos + [0, 3],
                               init_pos + [0, -3],
                               init_pos + [3, 0],
                               init_pos + [-3, 0]))

env = GPMultiAgent(navigation_map=nav,
                   movement_length=3,
                   number_of_agents=1,
                   initial_positions=init_pos,
                   initial_meas_locs=initial_meas_locs,
                   distance_budget=200,
                   device='cpu')

agent = DuelingDQNAgent(env,
                        memory_size=10000,
                        batch_size=64,
                        target_update=1,
                        soft_update=True,
                        tau=0.0001,
                        learning_starts=20,
                        gamma=0.99,
                        lr=1e-4,
                        prioritized_replay=True,
                        noisy=True
                        )


agent.train(episodes=10000)