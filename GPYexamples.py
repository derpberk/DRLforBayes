import numpy as np
import torch
from Environment.GPutils import GaussianProcessRegressorPytorch
from matplotlib import pyplot as plt

# Training data is 100 points in [0,1] inclusive regularly spaced
train_x = np.linspace(0, 10, 5000)
# True function is sin(2*pi*x) with Gaussian noise
train_y = np.sin(train_x ** 2)

test_x = np.linspace(0, 10, 10000)
test_y = np.sin(test_x ** 2)


# Initialize plot
f, ax = plt.subplots(1, 1, figsize=(4, 3))

model = GaussianProcessRegressorPytorch(training_iter=20, device='cpu', lengthscale=0.001)

model.fit(train_x, train_y)

mean, lower, upper = model.predict(test_x)

# Plot training data as black stars
ax.plot(train_x, train_y, 'k*')
# Plot predictive means as blue line
ax.plot(test_x, mean.cpu().numpy(), 'b')
# Shade between the lower and upper confidence bounds
ax.fill_between(test_x, lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5)
ax.set_ylim([-3, 3])
ax.legend(['Observed Data', 'Mean', 'Confidence'])
plt.show()