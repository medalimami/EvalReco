from .VirtualAlgo import VirtualAlgo
from typing import List, Tuple
import numpy as np
import pandas as pd

import lom


class AlgoBMF(VirtualAlgo):
    def __init__(self, rating_scale=(1, 5), dim=20):
        self.orm = None
        self.rating_scale = rating_scale
        self.dim = dim

    def fit(self, trainset: List[Tuple[int, int, float]]):
        self.orm = lom.Machine()

        self.num_users = 0
        self.num_movies = 0
        for u, i, r in trainset:
            if u > self.num_users:
                self.num_users = u
            if i > self.num_movies:
                self.num_movies = i
        self.num_users += 1
        self.num_movies += 1
        self.num_ratings = self.rating_scale[1] + 1
        X = np.zeros((self.num_users, self.num_movies * self.num_ratings), dtype=np.int8)

        for u, i, r in trainset:
            for j in range(self.num_ratings):
                if j <= r:
                    X[u, i * self.num_ratings + j] = 1
                else:
                    X[u, i * self.num_ratings + j] = -1

        data = self.orm.add_matrix(X, fixed=True)
        layer = self.orm.add_layer(latent_size=self.dim, child=data, model='OR-AND')
        self.orm.infer(burn_in_min=100, burn_in_max=1000, no_samples=50)

        # Construction de la matrice
        self.reconstructed_matrix = layer.output(technique='factor_map')

    def estimate(self, u: int, i: int) -> float:
        result = 0
        for j in range(self.num_ratings):
            # Vérifier si self.reconstructed_matrix est de la bonne dimension
            if u < self.reconstructed_matrix.shape[0] and i * self.num_ratings + j < self.reconstructed_matrix.shape[1]:
                result += self.reconstructed_matrix[u, i * self.num_ratings + j]
            else:
                return 0

        return result
    def get_base_model_name(self) -> str:
        return self.__class__.__name__