
from .VirtualAlgo import VirtualAlgo
import numpy as np
import copy
from typing import List, Tuple

class AlgoBaggingV2(VirtualAlgo):
    def __init__(self, base_model: VirtualAlgo, num_models: int, epsilon=0.05):
        self.base_model = base_model
        self.num_models = num_models
        self.epsilon = epsilon
        self.models = [copy.deepcopy(base_model) for _ in range(num_models)]

    def get_base_model_name(self):
        return self.base_model.__class__.__name__

    def fit(self, trainset: List[Tuple[int, int, float]]):
        weights = np.ones(len(trainset))
        
        for model in self.models:
            indices = np.random.choice(len(trainset), len(trainset), p=weights/np.sum(weights))
            model.fit([trainset[i] for i in indices])
            predictions = np.array([model.estimate(u, i) for u, i, _ in trainset])
            errors = np.abs(predictions - np.array([r for _, _, r in trainset]))
            weights = self.update_weights(weights, errors)
    
    def update_weights(self, weights, errors):
        #updated_weights = weights * np.exp(errors * self.epsilon)
        #updated_weights = self.epsilon + errors
        updated_weights = self.epsilon + errors*errors
        #updated_weights = weights * np.exp(errors + self.epsilon)
        #updated_weights = (errors > 0.5) + self.epsilon
        
        return updated_weights
            



    def estimate(self, u: int, i: int) -> float:
        predictions = [model.estimate(u, i) for model in self.models if hasattr(model, 'estimate')]
        assert len(predictions) > 0
        return np.median(predictions)
