
from .VirtualAlgo import VirtualAlgo
import pandas as pd
from surprise import SVD
from surprise import Dataset, Reader
from surprise import BaselineOnly
from typing import List, Tuple

class AlgoBase(VirtualAlgo):
    def __init__(self, rating_scale=(1, 5)):
        self.algo = BaselineOnly(verbose=False)
        self.reader = Reader(rating_scale=rating_scale)

    def fit(self, trainset: List[Tuple[int, int, float]]):
        df = pd.DataFrame(trainset, columns=['user_id', 'item_id', 'rating'])
        dataset = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], self.reader)
        trainingset = dataset.build_full_trainset()
        self.raw2inner_id_users = trainingset._raw2inner_id_users
        self.raw2inner_id_items = trainingset._raw2inner_id_items
        self.algo.fit(trainingset)

    def estimate(self, u: int, i: int) -> float:
        if u not in self.raw2inner_id_users:
            u = -1
        else:
            u = self.raw2inner_id_users[u]
            
        if i not in self.raw2inner_id_items:
            i = -1
        else:
            i = self.raw2inner_id_items[i]
            
        return self.algo.estimate(u, i)

    def get_base_model_name(self) -> str:
        return self.__class__.__name__
