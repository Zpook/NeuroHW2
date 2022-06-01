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


    def fit(self, X, epochs):
        n_samples, n_attributes = X.shape

        # Initialize the points randomly (weights)
        self.weights = torch.rand((self.n_points, n_attributes), dtype=torch.double)

        # From numpy conversion
        X = torch.from_numpy(X).type(torch.double)

        # Shuffling
        if self.allowShuffle:
            indices = torch.randperm(n_samples)
            X = X[indices, :]

        # Scaling W in the same range as X
        if self.allowScale:
            self.weights = self.weights * (torch.max(X) - torch.min(X)) + torch.min(X)

        # Record each W for each t (debugging)
        if self.allowHistory:
            self.history = self.weights.reshape(1, self.n_points, n_attributes)

        # The training loop
        for t in trange(epochs):
            x = X[t % n_samples, :]  # The current sampled x
            x_dists = x - self.weights  # Distances from x to W

            # Find the winning point
            distances = torch.pow((x_dists), 2).sum(axis=1)  # [n_points x 1]
            winner = torch.argmin(distances)

            # Lateral distance between neurons
            lat_dist = torch.pow((self.weights - self.weights[winner, :]), 2).sum(
                axis=1
            )  # [n_points x 1]

            # Update the learning rate
            alpha = self.alpha * exp(-t / self.t_alpha)  # [scalar]

            # Update the neighborhood size
            sigma = self.sigma * exp(-t / self.t_sigma)  # [scalar]

            # Evaluate the topological neighborhood
            h = torch.exp(-lat_dist / (2 * sigma**2)).reshape(
                (self.n_points, 1)
            )  # [n_points x 1]

            # Update W
            self.weights += alpha * h * (x_dists)

            if self.allowHistory:
                self.history = torch.cat(
                    (
                        self.history,
                        self.weights.reshape(1, self.n_points, n_attributes),
                    ),
                    axis=0,
                )
