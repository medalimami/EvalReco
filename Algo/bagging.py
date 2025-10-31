
from .VirtualAlgo import VirtualAlgo
import numpy as np
import copy
from typing import List, Tuple

class AlgoBagging(VirtualAlgo):
    def __init__(self, base_model, num_models):
        self.base_model = base_model
        self.num_models = num_models
        self.models = [copy.deepcopy(base_model) for _ in range(num_models)]

    def get_base_model_name(self):
        return self.base_model.__class__.__name__

    def fit(self, trainset: List[Tuple[int, int, float]]):
        for i, model in enumerate(self.models):
            # Selectionner 90% des données de trainset de maniere aléatoire pour l'entrainement de model
            np.random.shuffle(trainset)
            split_size = int(len(trainset) * 0.9)
            model.fit(trainset[:split_size])
            

    def estimate(self, u: int, i: int) -> float:
        predictions = []

        # Obtenir les prédictions de chaque modèle
        for model in self.models:
            prediction = model.estimate(u, i)
            predictions.append(prediction)

        assert predictions, "ERROR: No predictions available"
        
        # Calculer et retourner la médiane des prédictions
        return np.median(predictions)
