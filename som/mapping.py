import torch
from math import exp
from tqdm import trange


class SOM:
    def __init__(
        self,
        alpha0=0.5,
        t_alpha=25,
        sigma0=2,
        t_sigma=25,
        weights=None,
        scale=True,
        history=True,
        shuffle=True,
    ):

        self.alpha = alpha0
        self.alphaMaxTime = t_alpha
        self.sigma = sigma0
        self.sigmaMaxTime = t_sigma
        self.allowScale = scale
        self.allowHistory = history
        self.allowShuffle = shuffle

        if weights is None:
            raise Exception("Weights not set")

        self.weights = weights
        self.n_points = self.weights.shape[0]
        self.attributeNumber = self.weights.shape[1]


        # Initialize the points randomly (weights)
        # self.weights = torch.rand((self.n_points, n_attributes), dtype=torch.double)

    def ScalarUpdate_Power(self, value, currentTime, maxTime):
        return value * exp(-currentTime / maxTime)

    def Neighbourhood_Gaussian(self, lateralDistances, sigma):
        return torch.exp(-lateralDistances / (2 * sigma**2)).reshape(
            (self.n_points, 1)
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
            self.history = self.weights.reshape(1, self.n_points, n_attributes)

        # The training loop
        for epochIndex in trange(epochs):
            sample = input[epochIndex % n_samples, :]
            distances = sample - self.weights

            # Find the winning point
            euclideanDistances = torch.pow((distances), 2).sum(axis=1)
            BMUIndex = torch.argmin(euclideanDistances)
            BMU = self.weights[BMUIndex]

            # Lateral distance between neurons
            lateralDistances = torch.pow(
                (self.weights - BMU), 2
            ).sum(axis=1)

            # Update the learning rate
            alpha = self.ScalarUpdate_Power(self.alpha, epochIndex, self.alphaMaxTime)


            # Update the neighborhood size
            sigma = self.ScalarUpdate_Power(self.sigma, epochIndex, self.sigmaMaxTime)


            # Evaluate the topological neighborhood
            changeRate = self.Neighbourhood_Gaussian(lateralDistances,sigma)

            maxChange = 0.15
            changeRate[changeRate>maxChange] = maxChange
            changeRate[BMUIndex] = 1

            # Update weights
            self.weights += alpha * changeRate * (distances)

            if self.allowHistory:
                self.history = torch.cat(
                    (
                        self.history,
                        self.weights.reshape(1, self.n_points, n_attributes),
                    ),
                    axis=0,
                )
