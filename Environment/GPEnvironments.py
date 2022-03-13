import gym
import matplotlib.pyplot as plt
import numpy as np
from GPutils import GaussianProcessRegressorPytorch
from sklearn.metrics import mean_squared_error as MSE
from GroundTruths import ShekelGT

class DiscreteVehicle:

    def __init__(self, initial_position, n_actions, movement_length, navigation_map):

        self.initial_position = initial_position
        self.position = np.copy(initial_position)
        self.waypoints = np.expand_dims(np.copy(initial_position), 0)
        self.trajectory = np.copy(self.waypoints)

        self.distance = 0.0
        self.num_of_collisions = 0
        self.action_space = gym.spaces.Discrete(n_actions)
        self.angle_set = np.linspace(0, 2 * np.pi, n_actions, endpoint=False)
        self.movement_length = movement_length
        self.navigation_map = navigation_map

    def move(self, action):

        self.distance += self.movement_length
        angle = self.angle_set[action]
        movement = np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])

        next_position = np.clip(self.position + movement, (0,0), np.array(self.navigation_map.shape)-1)

        if self.check_collision(next_position):
            collide = True
            self.num_of_collisions += 1
        else:
            collide = False
            self.position = next_position
            self.waypoints = np.vstack((self.waypoints, [self.position]))

        return collide

    def check_collision(self, next_position):

        if self.navigation_map[int(next_position[0]), int(next_position[1])] == 0:
            return True
        return False


    @staticmethod
    def compute_trajectory_between_points(p1, p2):
        trajectory = None

        p = p1.astype(int)
        d = p2.astype(int) - p1.astype(int)
        N = np.max(np.abs(d))
        s = d / N

        for ii in range(0, N):
            p = p + s
            if trajectory is None:
                trajectory = np.array([np.rint(p)])
            else:
                trajectory = np.vstack((trajectory, [np.rint(p)]))

        return trajectory.astype(int)

    def reset(self, initial_position):

        self.initial_position = initial_position
        self.position = np.copy(initial_position)
        self.waypoints = np.expand_dims(np.copy(initial_position), 0)
        self.trajectory = np.copy(self.waypoints)
        self.distance = 0.0
        self.num_of_collisions = 0

    def check_action(self, action):

        angle = self.angle_set[action]
        movement = np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])
        next_position = self.position + movement

        return self.check_collision(next_position)

    def move_to_position(self, goal_position):

        """ Add the distance """
        assert self.navigation_map[goal_position[0], goal_position[1]] == 1, "Invalid position to move"
        self.distance += np.linalg.norm(goal_position - self.position)
        """ Update the position """
        self.position = goal_position


class DiscreteFleet:

    def __init__(self, number_of_vehicles, n_actions, initial_positions, movement_length, navigation_map):

        self.number_of_vehicles = number_of_vehicles
        self.initial_positions = initial_positions
        self.n_actions = n_actions
        self.movement_length = movement_length
        self.vehicles = [DiscreteVehicle(initial_position=initial_positions[k],
                                         n_actions=n_actions,
                                         movement_length=movement_length,
                                         navigation_map=navigation_map) for k in range(self.number_of_vehicles)]

        self.measured_values = None
        self.measured_locations = None
        self.fleet_collisions = 0

    def move(self, fleet_actions):

        collision_array = [self.vehicles[k].move(fleet_actions[k]) for k in range(self.number_of_vehicles)]

        self.fleet_collisions = np.sum([self.vehicles[k].num_of_collisions for k in range(self.number_of_vehicles)])

        return collision_array

    def measure(self, gt):

        """
        Take a measurement in the given N positions
        :param gt_field:
        :return: An numpy array with dims (N,2)
        """
        positions = np.array([self.vehicles[k].position for k in range(self.number_of_vehicles)])

        values = []
        for pos in positions:
            values.append(gt.evaluate(pos))

        if self.measured_locations is None:
            self.measured_locations = positions
            self.measured_values = values
        else:
            self.measured_locations = np.vstack((self.measured_locations, positions))
            self.measured_values = np.hstack((self.measured_values, values))

        return self.measured_values, self.measured_locations

    def reset(self, initial_positions):

        for k in range(self.number_of_vehicles):
            self.vehicles[k].reset(initial_position=initial_positions[k])

        self.measured_values = None
        self.measured_locations = None
        self.fleet_collisions = 0

    def get_distances(self):

        return [self.vehicles[k].distance for k in range(self.number_of_vehicles)]

    def check_collisions(self, test_actions):

        return [self.vehicles[k].check_action(test_actions[k]) for k in range(self.number_of_vehicles)]

    def move_fleet_to_positions(self, goal_list):
        """ Move the fleet to the given positions.
         All goal positions must ve valid. """

        goal_list = np.atleast_2d(goal_list)

        for k in range(self.number_of_vehicles):
            self.vehicles[k].move_to_position(goal_position=goal_list[k])


class GPMultiAgent(gym.Env):

    def __init__(self, navigation_map, number_of_agents, initial_positions, movement_length, distance_budget):

        self.navigation_map = navigation_map
        self.visitable_positions = np.column_stack(np.where(navigation_map == 1))
        self.number_of_agents = number_of_agents
        self.initial_positions = initial_positions
        self.distance_budget = distance_budget

        self.fleet = DiscreteFleet(number_of_vehicles=number_of_agents,
                                   n_actions=8,
                                   initial_positions=initial_positions,
                                   movement_length=movement_length,
                                   navigation_map=navigation_map)

        self.measured_locations = None
        self.measured_values = None
        self.mu = None
        self.lower_confidence = None
        self.upper_confidence = None
        self.GPR = GaussianProcessRegressorPytorch(training_iter=100)

        self.gt = ShekelGT(self.navigation_map.shape, self.visitable_positions)




    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # Process movements #
        collision_array = self.fleet.move(action)

        self.measured_values, self.measured_locations = self.fleet.measure(gt=self.gt)
        # Predict the new model #
        self.GPR.fit(self.measured_locations, self.measured_values)
        mu, lower_confidence, upper_confidence = self.GPR.predict(self.visitable_positions)
        # Reshape #
        self.mu = mu.cpu().numpy().reshape(self.navigation_map.shape)
        self.lower_confidence = lower_confidence.cpu().numpy().reshape(self.navigation_map.shape)
        self.upper_confidence = upper_confidence.cpu().numpy().reshape(self.navigation_map.shape)

        mse = MSE(y_true = self.gt.GroundTruth_field, y_pred=mu.cpu().numpy())

        # Compute reward #
        rewards = [-10 if collision_array[i] == 1 else mse for i in range(self.number_of_agents)]

        state = self.render_state()

        done = np.mean(self.fleet.get_distances()) > self.distance_budget

        return state, rewards, False, {}


    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """

        # Reset the positions #
        self.fleet.reset(self.initial_positions)

        self.gt.reset()

        # Take measurements
        self.measured_values, self.measured_locations = self.fleet.measure(gt=self.gt)
        # Predict the new model #
        self.GPR.fit(self.measured_locations, self.measured_values)
        mu, lower_confidence, upper_confidence = self.GPR.predict(self.visitable_positions)
        # Reshape #
        self.mu = mu.cpu().numpy().reshape(self.navigation_map.shape)
        self.lower_confidence = lower_confidence.cpu().numpy().reshape(self.navigation_map.shape)
        self.upper_confidence = upper_confidence.cpu().numpy().reshape(self.navigation_map.shape)

        state = self.render_state()

        return state


    def render_state(self):

        nav_map = np.copy(self.navigation_map)
        mean = np.copy(self.mu)
        uncertainty = self.upper_confidence - self.lower_confidence

        return np.dstack((nav_map, mean, uncertainty))

    def render(self, mode='human'):
            """Renders the environment.

            The set of supported modes varies per environment. (And some
            environments do not support rendering at all.) By convention,
            if mode is:

            - human: render to the current display or terminal and
              return nothing. Usually for human consumption.
            - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
              representing RGB values for an x-by-y pixel image, suitable
              for turning into a video.
            - ansi: Return a string (str) or StringIO.StringIO containing a
              terminal-style text representation. The text can include newlines
              and ANSI escape sequences (e.g. for colors).

            Note:
                Make sure that your class's metadata 'render.modes' key includes
                  the list of supported modes. It's recommended to call super()
                  in implementations to use the functionality of this method.

            Args:
                mode (str): the mode to render with

            Example:

            class MyEnv(Env):
                metadata = {'render.modes': ['human', 'rgb_array']}

                def render(self, mode='human'):
                    if mode == 'rgb_array':
                        return np.array(...) # return RGB frame suitable for video
                    elif mode == 'human':
                        ... # pop up a window and render
                    else:
                        super(MyEnv, self).render(mode=mode) # just raise an exception
            """
            raise NotImplementedError

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        return


if __name__ == '__main__':

    nav = np.ones((100,100))
    init_pos = np.random.rand(2,2) * 100
    env = GPMultiAgent(navigation_map=nav,
                       movement_length=6,
                       number_of_agents=2,
                       initial_positions=init_pos,
                       distance_budget=200)

    s = env.reset()

    while True:

        s,r,d,_ = env.step(np.random.randint(0,8,2))

        print("Reward: ", r)
        plt.imshow((s[:,:,1] - env.gt.GroundTruth_field.reshape()))
        plt.plot(env.fleet.vehicles[0].position[1],env.fleet.vehicles[0].position[0], 'xb')
        plt.plot(env.fleet.vehicles[1].position[1],env.fleet.vehicles[1].position[0], 'xr')
        plt.pause(0.5)
