import numpy as np
import gpytorch
import torch.optim
from utils import print_log, message_log


class ExactGPModel(gpytorch.models.ExactGP):
    """ Exact model gaussian regressor with RBF Kernel """
    def __init__(self, train_x, train_y, likelihood, lengthscale):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module.base_kernel.lengthscale = lengthscale
        self.num_outputs = 1

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcessRegressorPytorch:
    """ GP Regressor wrapper for train and evaluate """

    def __init__(self, training_iter=10, device = None, lengthscale = 10):

        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f'WARNING: Using {self.device} for GP Regression')

        self.y_train = None
        self.x_train = None
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=1E-5)
        self.GPmodel = ExactGPModel(train_x=self.x_train,
                                    train_y=self.y_train,
                                    likelihood=self.likelihood,
                                    lengthscale=lengthscale).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.GPmodel.parameters(), lr=0.1)
        # LogLikelihood
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.GPmodel)
        self.training_iter = training_iter



    def fit(self, x, y):

        self.x_train = torch.tensor(x, device=self.device, dtype=torch.float)
        self.y_train = torch.tensor(y, device=self.device, dtype=torch.float)

        self.GPmodel.set_train_data(inputs=self.x_train, targets=self.y_train, strict=False)

        # Train mode
        self.GPmodel.train()
        self.likelihood.train()

        for _ in range(self.training_iter):

            # Reset the gradients
            self.optimizer.zero_grad()
            # Predict the output
            output = self.GPmodel(self.x_train)
            # Compute Max. Likelihood
            loss = -self.mll(output, self.y_train)

            loss.backward()

            self.optimizer.step()


    def predict(self, x_eval):

        assert self.GPmodel is not None, message_log("ERROR: you have to run fit method first!", 'ERROR')

        # Eval mode #
        self.GPmodel.eval()
        self.likelihood.eval()

        x_eval_tensor = torch.tensor(x_eval, device=self.device, dtype=torch.float)

        # Fast prediction
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.GPmodel(x_eval_tensor))

        with torch.no_grad():

            # Lower & upper confidence bounds
            low, up = observed_pred.confidence_region()
            mean = observed_pred.mean


        return mean, low, up



if __name__ == '__main__':

    import matplotlib.pyplot as plt

    TYPE = '1D'

    if TYPE == '1D':

        x_true = np.linspace(0,10,100)
        y_true = np.sin(2*np.pi * (x_true/10))

        x_train = np.random.rand(10) * 10
        y_train = np.sin(2*np.pi * (x_train/10))

        GPR = GaussianProcessRegressorPytorch(training_iter=1000)

        GPR.fit(x_train, y_train)

        mean, low, up = GPR.predict(x_true)

        mean = mean.cpu().numpy()
        low = low.cpu().numpy()
        up = up.cpu().numpy()

        plt.plot(x_train, y_train, 'xk', label='Sampled vals')
        plt.plot(x_true, y_true, 'r--', label = 'True function')
        plt.plot(x_true, mean, 'b-', label = 'Predicted function')
        plt.fill_between(x_true, low, up, color='Blue', alpha=0.1)

        plt.legend()

        plt.show()

    else:


        def fun(x, y):
            return ((x-5)/2) ** 2 + (y-5)**2/2

        x1_true, x2_true = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))

        x_true = np.column_stack((x1_true.flatten(),x2_true.flatten()))

        zs = np.array(fun(np.ravel(x1_true), np.ravel(x2_true)))
        y_true = zs.reshape(x1_true.shape)

        x_train = np.random.rand(200, 2) * 10
        y_train = fun(x_train[:,0], x_train[:,1])

        GPR = GaussianProcessRegressorPytorch(training_iter=1000)

        GPR.fit(x_train, y_train)

        mean, low, up = GPR.predict(x_true)
        mean = mean.cpu().numpy().reshape(x1_true.shape)
        low = low.cpu().numpy()
        up = up.cpu().numpy()

        plt.matshow(mean)
        plt.colorbar()
        plt.matshow(y_true)
        plt.colorbar()
        plt.show()

























