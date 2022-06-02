import matplotlib.pyplot as plt
import numpy as np
import torch
from ShapeGen import ShapeGen

from ShapeGen import Rectangle, ShapeGen
from som.mapping import SOM


SHOW_HISTORY = False
HISTORY_STEPSIZE = 100
SHOW_OUTPUT = True
SHOW_INPUT = False

USE_EXAMPLE_DATA = False


x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)
XX, YY = np.meshgrid(x, y)

data = np.vstack([XX.reshape(-1), YY.reshape(-1)]).transpose()


palmShape = Rectangle((0.15,0.15),(0.85,0.35))
finger1 = Rectangle((0.15,0.25),(0.25,0.65))
finger2 = Rectangle((0.35,0.35),(0.45,0.75))
finger3 = Rectangle((0.55,0.35),(0.65,0.85))
finger4 = Rectangle((0.75,0.35),(0.85,0.75))

# allShapes = [palmShape]
allShapes = [palmShape,finger1,finger2,finger3,finger4]

shapeFilter = ShapeGen(allShapes)

data = np.array(shapeFilter.FilterPoints(data))

x = np.linspace(0, 1, 15)
y = np.linspace(0, 1, 15)
XX, YY = np.meshgrid(x, y)

weights = np.vstack([XX.reshape(-1), YY.reshape(-1)]).transpose()
weights = torch.tensor(weights)

gridCoords = weights.reshape(15,15,2)

if USE_EXAMPLE_DATA:

    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA

    dataset = load_iris()
    train = dataset.data
    pca = PCA(n_components=2)
    train_pca = pca.fit_transform(train)
    data = train_pca


model = SOM(
    alpha0=0.1,
    t_alpha=5000,
    sigma0=0.1,
    t_sigma=1000,
    weights=weights,
    scale=True,
    history=True,
)

model.fit(data,5000)

if SHOW_OUTPUT:
    weights = model.weights

    fig, ax = plt.subplots()
    fig.suptitle("Train set (PCA-reduced) and weights")

    t = ax.scatter(data[:, 0], data[:, 1])
    w = ax.scatter(weights[:, 0], weights[:, 1])

    fig.legend((t, w), ("Train", "Weights"))
    plt.xlim((0,1))
    plt.ylim((0,1))
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
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.show()
