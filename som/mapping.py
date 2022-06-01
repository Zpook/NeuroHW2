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
        history=False,
    ):

        self.alpha = alpha0
        self.t_alpha = t_alpha
        self.sigma = sigma0
        self.t_sigma = t_sigma
        self.allowScale = scale
        self.allowHistory = history

        if weights is None:
            self.weights = torch.rand((500, 2), dtype=torch.double)

        else:
            self.weights = weights

        self.n_points = self.weights.shape[0]
        self.attributeNumber = self.weights.shape[1]

        self.params = {
            "n_points": 500,
            "alpha0": 0.5,
            "t_alpha": 25,
            "sigma0": 2,
            "t_sigma": 25,
            "epochs": 300,
            "seed": 124,
            "scale": True,
            "shuffle": True,
            "history": True,
        }

    def fit_old(self, X):
        n_samples, n_attributes = X.shape

        # Initialize the points randomly (weights)
        self.W = torch.rand((self.params["n_points"], n_attributes), dtype=torch.double)

        # From numpy conversion
        X = torch.from_numpy(X).type(torch.double)

        # Shuffling
        if self.params["shuffle"]:
            indices = torch.randperm(n_samples)
            X = X[indices, :]

        # Scaling W in the same range as X
        if self.params["scale"]:
            self.W = self.W * (torch.max(X) - torch.min(X)) + torch.min(X)

        # Record each W for each t (debugging)
        if self.params["history"]:
            self.history = self.W.reshape(1, self.params["n_points"], n_attributes)

        # The training loop
        for t in trange(self.params["epochs"]):
            x = X[t % n_samples, :]  # The current sampled x
            x_dists = x - self.W  # Distances from x to W

            # Find the winning point
            distances = torch.pow((x_dists), 2).sum(axis=1)  # [n_points x 1]
            winner = torch.argmin(distances)

            # Lateral distance between neurons
            lat_dist = torch.pow((self.W - self.W[winner, :]), 2).sum(
                axis=1
            )  # [n_points x 1]

            # Update the learning rate
            alpha = self.params["alpha0"] * exp(-t / self.params["t_alpha"])  # [scalar]

            # Update the neighborhood size
            sigma = self.params["sigma0"] * exp(-t / self.params["t_sigma"])  # [scalar]

            # Evaluate the topological neighborhood
            h = torch.exp(-lat_dist / (2 * sigma**2)).reshape(
                (self.params["n_points"], 1)
            )  # [n_points x 1]

            # Update W
            self.W += alpha * h * (x_dists)

            if self.params["history"]:
                self.history = torch.cat(
                    (
                        self.history,
                        self.W.reshape(1, self.params["n_points"], n_attributes),
                    ),
                    axis=0,
                )

    def fit(self, input, epochs):
        n_samples, n_attributes = X.shape

        # Initialize the points randomly (weights)
        self.W = torch.rand((self.params["n_points"], n_attributes), dtype=torch.double)

        # From numpy conversion
        X = torch.from_numpy(X).type(torch.double)

        # Shuffling
        if self.params["shuffle"]:
            indices = torch.randperm(n_samples)
            X = X[indices, :]

        # Scaling W in the same range as X
        if self.params["scale"]:
            self.W = self.W * (torch.max(X) - torch.min(X)) + torch.min(X)

        # Record each W for each t (debugging)
        if self.params["history"]:
            self.history = self.W.reshape(1, self.params["n_points"], n_attributes)

        # The training loop
        for t in trange(self.params["epochs"]):
            x = X[t % n_samples, :]  # The current sampled x
            x_dists = x - self.W  # Distances from x to W

            # Find the winning point
            distances = torch.pow((x_dists), 2).sum(axis=1)  # [n_points x 1]
            winner = torch.argmin(distances)

            # Lateral distance between neurons
            lateralDistance = torch.pow((self.W - self.W[winner, :]), 2).sum(
                axis=1
            )  # [n_points x 1]

            # Update the learning rate
            alpha = self.params["alpha0"] * exp(-t / self.params["t_alpha"])  # [scalar]

            # Update the neighborhood size
            sigma = self.params["sigma0"] * exp(-t / self.params["t_sigma"])  # [scalar]

            # Evaluate the topological neighborhood
            h = torch.exp(-lateralDistance / (2 * sigma**2)).reshape(
                (self.params["n_points"], 1)
            )  # [n_points x 1]

            # Update W
            self.W += alpha * h * (x_dists)

            if self.params["history"]:
                self.history = torch.cat(
                    (
                        self.history,
                        self.W.reshape(1, self.params["n_points"], n_attributes),
                    ),
                    axis=0,
                )
