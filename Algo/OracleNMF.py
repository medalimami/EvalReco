from .VirtualOracle import VirtualOracle
from typing import List, Tuple

import numpy as np
import math
import random

# This Oracle uses a Non-negative Matrix Factorization to generate the ratings
class OracleNMF(VirtualOracle):
    def __init__(self, num_users, num_items, rank=10, max_rating=5.0):
        self.num_users = num_users
        self.num_items = num_items
        self.rank = rank
        
        maxIndividuel = math.sqrt(max_rating / rank)
        
        self.A = np.random.rand(num_users, rank)*maxIndividuel
        self.B = np.random.rand(rank, num_items)*maxIndividuel


    def get_rating(self, num_users, num_items) -> float:
        return np.dot(self.A[num_users], self.B[:, num_items])
    
    def get_users(self):
        users = set()
        for i in range(self.num_users):
            users.add(i)
        return users
    
    def get_items(self):
        items = set()
        for i in range(self.num_items):
            items.add(i)
        return items
    
    def get_random_exemple(self) -> List[Tuple[int, int, float]]:
        idu = random.randint(0, self.num_users-1)
        idi = random.randint(0, self.num_items-1)
        return [idu, idi, self.get_rating(idu, idi)]
    
    def get_number_of_ratings(self) -> int:
        return self.num_users * self.num_items

# In the case of a small matrix, we can generate the matrix at once
class OracleNMF_small(VirtualOracle):
    def __init__(self, num_users, num_items, rank=10, max_rating=5.0):
        self.num_users = num_users
        self.num_items = num_items
        self.rank = rank
        
        maxIndividuel = math.sqrt(max_rating / rank)
        
        A = np.random.rand(num_users, rank)*maxIndividuel
        B = np.random.rand(rank, num_items)*maxIndividuel
        self.M = np.dot(A, B)

    def get_rating(self, id_user: int, id_item: int):
        return self.M[id_user, id_item]
    
    def get_users(self) -> List[int]:
        users = set()
        for i in range(self.num_users):
            users.add(i)
        return users
    
    def get_items(self) -> List[int]:
        items = set()
        for i in range(self.num_items):
            items.add(i)
        return items

    def get_random_exemple(self) -> List[Tuple[int, int, float]]:
        idu = random.randint(0, self.num_users-1)
        idi = random.randint(0, self.num_items-1)
        return [idu, idi, self.get_rating(idu, idi)]
    
    def get_number_of_ratings(self) -> int:
        return self.num_users * self.num_items