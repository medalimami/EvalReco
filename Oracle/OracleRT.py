
from .VirtualOracle import VirtualOracle
import random
import numpy as np
from typing import List, Tuple

class OracleRandomTrees:
    def __init__(self, num_users, num_items, num_features, leaf_probability=0.5):
        self.num_users = num_users
        self.num_items = num_items
        self.num_features = num_features
        self.leaf_probability = leaf_probability
        self.features_list = [str(i) for i in range(1, num_features + 1)]
        self.feature_values = self.generate_fixed_features()  # Caractéristiques fixes pour les films
        self.ratings_matrix = self.generate_ratings_matrix()

    def create_tree(self, features_list, leaf_probability=0.5):
        if not features_list or random.random() < leaf_probability:
            return random.randint(1, 5)

        genre = random.choice(features_list)
        features_list.remove(genre)

        tree = {"feature": genre, "branches": {}}

        for i in range(2):
            if random.random() < leaf_probability:
                tree["branches"][f"Branch {i}"] = random.randint(1, 5)
            else:
                tree["branches"][f"Branch {i}"] = self.create_tree(features_list, leaf_probability)
        return tree

    def generate_fixed_features(self):
        return {i: {str(f): random.randint(0, 1) for f in range(1, self.num_features + 1)} for i in range(self.num_items)}

    def generate_ratings_matrix(self):
        ratings_matrix = np.zeros((self.num_users, self.num_items))

        for u in range(self.num_users):
            random_tree = self.create_tree(self.features_list.copy(), self.leaf_probability)
            for i in range(self.num_items):
                feature_values = self.feature_values[i]  # Utilisation des caractéristiques fixes
                ratings_matrix[u, i] = self.calculate_tree_rating(random_tree, feature_values)

        return ratings_matrix

    def calculate_tree_rating(self, tree, feature_values):
        if isinstance(tree, int):
            return tree
        feature_name = tree["feature"]
        branch = feature_values[feature_name]
        return self.calculate_tree_rating(tree["branches"][f"Branch {branch}"], feature_values)

    def get_rating(self, id_user: int, id_item: int) -> float:
        return self.ratings_matrix[id_user, id_item]

    def get_users(self) -> List[int]:
        return set(range(self.num_users))

    def get_items(self) -> List[int]:
        return set(range(self.num_items))

    def get_random_exemple(self) -> List[Tuple[int, int, float]]:
        user = random.randint(0, self.num_users - 1)
        item = random.randint(0, self.num_items - 1)
        rating = self.get_rating(user, item)
        return [user, item, rating]

    def get_number_of_ratings(self) -> int:
        return self.num_users * self.num_items
