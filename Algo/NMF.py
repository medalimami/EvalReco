from .VirtualAlgo import VirtualAlgo
import pandas as pd
from surprise import NMF
from surprise import Dataset, Reader
from typing import List, Tuple


class AlgoNMF(VirtualAlgo):
    def __init__(self, rating_scale=(1, 5), dim=20):
        self.algo = NMF(n_factors=dim, biased=False)
        self.reader = Reader(rating_scale=rating_scale)
        self.rating_scale = rating_scale

    def fit(self, trainset: List[Tuple[int, int, float]]):
        df = pd.DataFrame(trainset, columns=['user_id', 'item_id', 'rating'])
        dataset = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], self.reader)
        trainingset = dataset.build_full_trainset()
        self.raw2inner_id_users = trainingset._raw2inner_id_users
        self.raw2inner_id_items = trainingset._raw2inner_id_items
        self.algo.fit(trainingset)
        self.user_mean_ratings = df.groupby('user_id')['rating'].mean().to_dict()
        self.item_mean_ratings = df.groupby('item_id')['rating'].mean().to_dict()

    def estimate(self, u: int, i: int) -> float:
        if u not in self.raw2inner_id_users:
            return self.user_mean_ratings.get(u, (self.rating_scale[0] + self.rating_scale[1]) / 2)
        else:
            u = self.raw2inner_id_users[u]

        if i not in self.raw2inner_id_items:
            return self.item_mean_ratings.get(i, (self.rating_scale[0] + self.rating_scale[1]) / 2)

        else:
            i = self.raw2inner_id_items[i]

        est = self.algo.estimate(u, i)
        if est >= self.rating_scale[0] and est <= self.rating_scale[1]:
            return est
        elif est >= self.rating_scale[1]:
            return self.rating_scale[1]
        else:
            return self.rating_scale[0]
        #return self.algo.estimate(u, i) if self.algo.estimate(u, i) >=1 and self.algo.estimate(u, i) <= 5 else self.algo.estimate(u, i)

    def get_base_model_name(self) -> str:
        return self.__class__.__name__