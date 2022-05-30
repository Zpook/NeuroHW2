import matplotlib.pyplot as plt
import numpy as np
import torch

from som.mapping import SOM


x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
XX, YY = np.meshgrid(x, y)

data = np.vstack([XX.reshape(-1), YY.reshape(-1)]).transpose()

x = np.linspace(0, 1, 15)
y = np.linspace(0, 1, 15)
XX, YY = np.meshgrid(x, y)

weights = np.vstack([XX.reshape(-1), YY.reshape(-1)]).transpose()
weights = torch.tensor(weights)

model = SOM(
    alpha0=0.5,
    t_alpha=25,
    sigma0=2,
    t_sigma=25,
    attribNum=2,
    weights=weights,
    scale=True,
    history=True,
)
model.fit(data, 300)

weights = model.get_weights()

# Plot the train dataset and the weights
historyNum = model.history.shape[0]
for historyIndex in range(historyNum):
    weights = model.history[historyIndex, :, :]
    fig, ax = plt.subplots()
    fig.suptitle("Train set (PCA-reduced) and weights")
    t = ax.scatter(data[:, 0], data[:, 1])
    w = ax.scatter(weights[:, 0], weights[:, 1])
    fig.legend((t, w), ("Train", "Weights"))
    plt.show()
