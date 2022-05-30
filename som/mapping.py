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
        attribNum=2,
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

        self.weights = weights
        self.n_points = weights.shape[0]
        self.attributeNumber = weights.shape[1]

    def fit(self, X, epochs):
        n_samples, n_attributes = X.shape

        # From numpy conversion
        X = torch.from_numpy(X).type(torch.double)

        # Scaling W in the same range as X
        if self.allowScale:
            self.weights = self.weights * (torch.max(X) - torch.min(X)) + torch.min(X)

        # Record each W for each t (debugging)
        if self.allowHistory:
            self.history = self.weights.reshape(1, self.n_points, self.attributeNumber)

        # The training loop
        for t in trange(epochs):
            x = X[t % n_samples, :]  # The current sampled x
            x_dists = x - self.weights  # Distances from x to W

            # Find the winning point
            distances = torch.pow((x_dists), 2).sum(axis=1)  # [n_points x 1]
            minDistance = torch.argmin(distances)

            # Lateral distance between neurons
            lateralDistance = torch.pow((self.weights - self.weights[minDistance, :]), 2).sum(
                axis=1
            )  # [n_points x 1]

            # Update the learning rate
            alpha = self.alpha * exp(-t / self.t_alpha)  # [scalar]

            # Update the neighborhood size
            sigma = self.sigma * exp(-t / self.t_sigma)  # [scalar]

            # Evaluate the topological neighborhood
            h = torch.exp(-lateralDistance / (2 * sigma**2)).reshape(
                (self.n_points, 1)
            )  # [n_points x 1]

            # Update W
            self.weights += alpha * h * (x_dists)

            if self.allowHistory:
                self.history = torch.cat(
                    (
                        self.history,
                        self.weights.reshape(1, self.n_points, self.attributeNumber),
                    ),
                    axis=0,
                )

    def adjacency_matrix(self, M):
        M = torch.from_numpy(M)
        n_samples, n_attributes = M.shape

        # Broadcast the M matrix to a tensor of the shape
        # (n_sample, (n_samples, n_attributes))

        tensor = M.repeat(n_samples, 1, 1)  # Make n_samples copy of M
        M_flat = M.reshape(
            n_samples, 1, n_attributes
        )  # Each row of M or each copy in tensor
        distances = torch.pow((tensor - M_flat), 2).sum(axis=2).sqrt()
        return distances

    def get_weights(self):
        return self.weights
