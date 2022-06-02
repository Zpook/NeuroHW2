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
        self.t_alpha = t_alpha
        self.sigma = sigma0
        self.t_sigma = t_sigma
        self.allowScale = scale
        self.allowHistory = history
        self.allowShuffle = shuffle

        if weights is None:
            raise Exception("Weights not set")

        self.weights = weights
        self.n_points = self.weights.shape[0]
        self.attributeNumber = self.weights.shape[1]


    def fit(self, input, epochs):
        n_samples, n_attributes = input.shape

        # Initialize the points randomly (weights)
        self.weights = torch.rand((self.n_points, n_attributes), dtype=torch.double)

        # From numpy conversion
        input = torch.from_numpy(input).type(torch.double)

        # Shuffling
        if self.allowShuffle:
            indices = torch.randperm(n_samples)
            input = input[indices, :]

        # Scaling W in the same range as X
        if self.allowScale:
            self.weights = self.weights * (torch.max(input) - torch.min(input)) + torch.min(input)

        # Record each W for each t (debugging)
        if self.allowHistory:
            self.history = self.weights.reshape(1, self.n_points, n_attributes)

        # The training loop
        for t in trange(epochs):
            sample = input[t % n_samples, :]  # The current sampled x
            distances = sample - self.weights  # Distances from x to W

            # Find the winning point
            euclideanDistances = torch.pow((distances), 2).sum(axis=1)  # [n_points x 1]
            winner = torch.argmin(euclideanDistances)

            # Lateral distance between neurons
            lateralDistances = torch.pow((self.weights - self.weights[winner, :]), 2).sum(
                axis=1
            )  # [n_points x 1]

            # Update the learning rate
            alpha = self.alpha * exp(-t / self.t_alpha)  # [scalar]

            # Update the neighborhood size
            sigma = self.sigma * exp(-t / self.t_sigma)  # [scalar]

            # Evaluate the topological neighborhood
            h = torch.exp(-lateralDistances / (2 * sigma**2)).reshape(
                (self.n_points, 1)
            )  # [n_points x 1]

            # Update W
            self.weights += alpha * h * (distances)

            if self.allowHistory:
                self.history = torch.cat(
                    (
                        self.history,
                        self.weights.reshape(1, self.n_points, n_attributes),
                    ),
                    axis=0,
                )
