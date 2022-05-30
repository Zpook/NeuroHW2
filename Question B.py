import matplotlib.pyplot as plt
import numpy as np

from som.mapping import SOM


x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
XX, YY = np.meshgrid(x, y)

data = np.vstack([ XX.reshape(-1), YY.reshape(-1) ]).transpose()

parameters = {'n_points'  : 500,
              'alpha0'    : 0.5,
              't_alpha'   : 25,
              'sigma0'    : 2,
              't_sigma'   : 25,
              'epochs'    : 300,
              'seed'      : 124,
              'scale'     : True,
              'shuffle'   : True,
              'history'   : True}

# Load and train the model
model = SOM()
model.set_params(parameters)
model.fit(data)

weights = model.get_weights()

# Plot the train dataset and the weights
historyNum = model.history.shape[0]
for historyIndex in range(historyNum):
    weights = model.history[historyIndex,:,:]
    fig, ax = plt.subplots()
    fig.suptitle("Train set (PCA-reduced) and weights")
    t = ax.scatter(data[:,0], data[:,1])
    w = ax.scatter(weights[:, 0], weights[:, 1])
    fig.legend((t, w), ("Train", "Weights"))
    plt.show()