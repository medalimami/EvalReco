from .VirtualOracle import VirtualOracle
from surprise import Dataset
import math
import random
from typing import List, Tuple

class OracleSurpriseDataset(VirtualOracle):
    
    def __init__(self, dataName):
        self.dataset_name = dataName
        tmp = Dataset.load_builtin(dataName)
        self.data = {}
        self.users = set()
        self.items = set()
        for it in tmp.raw_ratings:
            self.users.add(int(it[0]))
            self.items.add(int(it[1]))
            self.data[int(it[0]), int(it[1])] = float(it[2])
        self.keys = list(self.data.keys())

    def get_rating(self, id_user: int, id_item: int) -> float:
        if (id_user, id_item) not in self.data:
            return math.nan
        return self.data[id_user, id_item]

    def get_users(self) -> List[int]:
        return self.users
    
    def get_items(self) -> List[int]:
        return self.items
    
    def get_random_exemple(self) -> List[Tuple[int, int, float]]:
        k = random.choice(self.keys)
        return [k[0], k[1], self.data[k]]

    def get_number_of_ratings(self) -> int:
        return len(self.data)