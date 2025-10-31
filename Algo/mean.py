from .VirtualAlgo import VirtualAlgo
import pandas as pd
from surprise import Dataset, Reader
from typing import List, Tuple


class AlgoMean(VirtualAlgo):
    def __init__(self, rating_scale=(1, 5), dim=20):
        self.reader = Reader(rating_scale=rating_scale)
        self.rating_scale = rating_scale
        self.user_mean_ratings = {}
        self.item_mean_ratings = {}

    def fit(self, trainset: List[Tuple[int, int, float]]):
        df = pd.DataFrame(trainset, columns=['user_id', 'item_id', 'rating'])
        dataset = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], self.reader)
        trainingset = dataset.build_full_trainset()
        self.raw2inner_id_users = trainingset._raw2inner_id_users
        self.raw2inner_id_items = trainingset._raw2inner_id_items

        self.item_mean_ratings = df.groupby('item_id')['rating'].mean().to_dict()

    def estimate(self, u: int, i: int) -> float:
        if i in self.item_mean_ratings:
            return round(self.item_mean_ratings[i])
        else:
            return 2.0

    def get_base_model_name(self) -> str:
        return self.__class__.__name__