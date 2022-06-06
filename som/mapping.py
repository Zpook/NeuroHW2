import torch
import numpy as np
from math import exp
from tqdm import trange
import matplotlib.pyplot as plt



class SOM:
    def __init__(
        self,
        nFeatures,
        gridShape,
        weights,
        alpha0=0.5,
        t_alpha=25,
        sigma0=2,
        t_sigma=25,
        scale=True,
        history=True,
        shuffle=True,
    ):

        self.nFeatures = nFeatures
        self.gridShape = gridShape
        self.weights = weights
        self.nWeights = self.weights.shape[0]

        self._unitGrid = self._GenerateUnitGrid()

        self.alpha = alpha0
        self.alphaMaxTime = t_alpha
        self.sigma = sigma0
        self.sigmaMaxTime = t_sigma
        self.allowScale = scale
        self.allowHistory = history
        self.allowShuffle = shuffle


    def _GenerateUnitGrid(self):
        x = np.arange(self.gridShape[0])
        y = np.arange(self.gridShape[1])
        XX, YY = np.meshgrid(x, y)

        unitGrid = np.vstack([XX.reshape(-1), YY.reshape(-1)]).transpose().reshape(self.gridShape[0],self.gridShape[1],self.nFeatures)
        unitGrid = torch.tensor(unitGrid)

        return unitGrid

    def ScalarUpdate(self, value, currentTime, maxTime):
        return value * exp(-currentTime / maxTime)

    def GuassianNeighbourhood(self, BMUGridCoords, sigma):
        
        distanceMatrix = torch.zeros(self.gridShape)

        distanceMatrix = torch.abs((self._unitGrid - torch.Tensor(BMUGridCoords)))
        distanceMatrix = distanceMatrix.sum(axis=2)
        distanceMatrix = torch.pow(distanceMatrix,2)

        return torch.exp(-distanceMatrix / (2 * sigma**2)).reshape(
            (self.nWeights, 1)
        )


    def fit(self, input, epochs):
        n_samples, n_attributes = input.shape

        # From numpy conversion
        input = torch.from_numpy(input).type(torch.double)

        # Shuffling
        if self.allowShuffle:
            indices = torch.randperm(n_samples)
            input = input[indices, :]

        # Scaling W in the same range as X
        if self.allowScale:
            self.weights = self.weights * (
                torch.max(input) - torch.min(input)
            ) + torch.min(input)

        # Record each W for each t (debugging)
        if self.allowHistory:
            self.history = self.weights.reshape(1, self.nWeights, n_attributes)

        # The training loop
        for epochIndex in trange(epochs):
            sample = input[epochIndex % n_samples, :]
            distances = sample - self.weights

            # Find the winning point
            euclideanDistances = torch.pow((distances), 2).sum(axis=1)
            BMUIndex = torch.argmin(euclideanDistances)
            BMUGridCoords = (int(BMUIndex/self.gridShape[0]),int(BMUIndex%self.gridShape[0]))

            # Update the learning rate
            alpha = self.ScalarUpdate(self.alpha, epochIndex, self.alphaMaxTime)

            # Update the neighborhood size
            sigma = self.ScalarUpdate(self.sigma, epochIndex, self.sigmaMaxTime)

            # Evaluate the topological neighborhood
            changeRate = self.GuassianNeighbourhood(BMUGridCoords,sigma)

            # Update weights
            self.weights += alpha * changeRate * (distances)

            if self.allowHistory:
                self.history = torch.cat(
                    (
                        self.history,
                        self.weights.reshape(1, self.nWeights, n_attributes),
                    ),
                    axis=0,
                )
