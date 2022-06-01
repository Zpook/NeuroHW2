import matplotlib.pyplot as plt
import numpy as np
import torch

from som.mapping import SOM


SHOW_HISTORY = False
HISTORY_STEPSIZE = 10
SHOW_OUTPUT = True

USE_EXAMPLE_DATA = False


x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
XX, YY = np.meshgrid(x, y)

data = np.vstack([XX.reshape(-1), YY.reshape(-1)]).transpose()

x = np.linspace(0, 1, 15)
y = np.linspace(0, 1, 15)
XX, YY = np.meshgrid(x, y)

weights = np.vstack([XX.reshape(-1), YY.reshape(-1)]).transpose()
weights = torch.tensor(weights)


if USE_EXAMPLE_DATA:

    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA

    dataset = load_iris()
    train = dataset.data
    pca = PCA(n_components=2)
    train_pca = pca.fit_transform(train)
    data = train_pca

model = SOM(
    alpha0=0.5,
    t_alpha=25,
    sigma0=2,
    t_sigma=25,
    weights=weights,
    scale=True,
    history=True,
)


model.fit(data,500)


if SHOW_OUTPUT:
    weights = model.W

    fig, ax = plt.subplots()
    fig.suptitle("Train set (PCA-reduced) and weights")

    t = ax.scatter(data[:, 0], data[:, 1])
    w = ax.scatter(weights[:, 0], weights[:, 1])

    fig.legend((t, w), ("Train", "Weights"))
    plt.show()


if SHOW_HISTORY:

    # Plot the train dataset and the weights
    historyNum = model.history.shape[0]

    for historyIndex in range(0,historyNum,HISTORY_STEPSIZE):
        weights = model.history[historyIndex, :, :]

        fig, ax = plt.subplots()
        fig.suptitle("Train set (PCA-reduced) and weights")

        t = ax.scatter(data[:, 0], data[:, 1])
        w = ax.scatter(weights[:, 0], weights[:, 1])

        fig.legend((t, w), ("Train", "Weights"))
        plt.show()
