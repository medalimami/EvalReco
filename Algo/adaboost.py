from .VirtualAlgo import VirtualAlgo
import numpy as np
import copy
from typing import List, Tuple

class AlgoAdaBoost(VirtualAlgo):
    def __init__(self, base_model: VirtualAlgo, num_models: int):
        self.base_model = base_model
        self.num_models = num_models
        self.models = []
        self.alphas = []

    def fit(self, trainset: List[Tuple[int, int, float]]):
        n = len(trainset)
        weights = np.ones(n) / n

        for i in range(self.num_models):
            model = copy.deepcopy(self.base_model)
            indices = np.random.choice(n, n, p=weights)
            training_subset = [trainset[i] for i in indices]
            model.fit(training_subset)

            predictions = np.array([model.estimate(u, i) for u, i, _ in trainset])
            errors = np.abs(predictions - np.array([r for _, _, r in trainset]))
            err = np.sum(weights * errors)

            #epsilon = 1e-10
            alpha = 0.5 * np.log((1 - err) / (err))
            self.alphas.append(alpha)
            self.models.append(model)

            weights *= np.exp(alpha * errors)
            weights /= np.sum(weights)

    def estimate(self, u: int, i: int) -> float:
        predictions = []
        for model in self.models:
            assert hasattr(model, 'estimate')
            prediction = model.estimate(u, i)
            predictions.append(prediction)
        assert len(predictions) > 0

        totalWeight = np.sum(self.alphas)
        tmp = [ (predictions[i], self.alphas[i]) for i in range(len(predictions))]
        tmp.sort(key=lambda x: x[0])
        
        tmpWeight = 0
        for i in range(len(tmp)):
            tmpWeight += tmp[i][1]
            if tmpWeight >= totalWeight/2:
                return tmp[i][0]
        assert False
        #return np.sign(weighted_sum)

