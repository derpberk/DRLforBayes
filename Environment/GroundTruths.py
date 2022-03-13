from deap.benchmarks import shekel
import numpy as np

class ShekelGT:

        def __init__(self, map_lims, visitable_positions):

            self.lims = map_lims

            self.A = None
            self.C = None
            self.visitable_positions = visitable_positions

            self.reset()

        def reset(self):
            number_of_peaks = np.random.randint(1, 6)
            self.pA = np.random.rand(number_of_peaks, 2)
            self.pC = np.ones(number_of_peaks) * 0.1
            number_of_peaks = np.random.randint(1, 6)
            self.nA = np.random.rand(number_of_peaks, 2)
            self.nC = np.ones(number_of_peaks) * 0.1

            self.GroundTruth_field = np.array(list(map(self.evaluate, self.visitable_positions)))

        def shekel_arg0(self, sol):

            value = shekel(np.asarray(sol)/self.lims, self.pA, self.pC)[0] - shekel(np.asarray(sol)/self.lims, self.nA, self.nC)[0]
            return value

        def evaluate(self, position):
            return self.shekel_arg0(position)



if __name__ == '__main__':

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.colors import LogNorm
    import matplotlib.pyplot as plt

    gt = ShekelGT((50,60))

    fig = plt.figure()
    # ax = Axes3D(fig, azim = -29, elev = 50)
    ax = Axes3D(fig)
    X = np.linspace(0, 50, 100)
    Y = np.linspace(0, 60, 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.fromiter(map(gt.evaluate, zip(X.flat, Y.flat)), dtype=float, count=X.shape[0] * X.shape[1]).reshape(
        X.shape)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)

    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()