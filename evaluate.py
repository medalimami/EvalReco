import numpy as np
import math
import random


class Mean:
    def __init__(self, title=""):
        self.nb = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.val_t = 3.0  # Pour le % de certitude (1 => 68; 2 => 95% ; 3 => 99,7)
        self.title = title

    def clear(self):
        self.nb = 0
        self.mean = 0.0
        self.M2 = 0.0

    def setT(self, t):
        self.val_t = t

    def add(self, val):
        self.nb += 1
        delta = val - self.mean
        self.mean += delta / self.nb
        self.M2 += delta * (val - self.mean)

    def getMeanMin(self):
        return self.getMean() - self.val_t * (self.getEcartType() / math.sqrt(self.nb))

    def getMeanMax(self):
        return self.getMean() + self.val_t * (self.getEcartType() / math.sqrt(self.nb))

    def getMean(self):
        return self.mean

    def getVariance(self):
        return self.M2 / (self.nb - 1.0)

    def getEcartType(self):
        return math.sqrt(self.getVariance())

    def size(self):
        return self.nb

    def print(self, unite=""):
        print(f"Average {self.title}: {self.getMean()}{unite} +- {(self.getMeanMax() - self.getMean())}{unite}")


def evaluate(algo, oracle, percentage_of_ratings_for_training, noise, seed=0):
    """
    Evaluate the given algorithm on the given oracle with Gaussian noise.
    :param algo: The algorithm to evaluate.
    :param oracle: The oracle to evaluate on.
    :param percentage_of_ratings_for_training: The percentage of ratings to generate for training.
    :param noise: The standard deviation of the Gaussian noise to add to the ratings (default is 0.0 for no noise).
    """

    if seed != 0:
        np.random.seed(seed)
        random.seed(seed)

    used = set()

    number_of_ratings_for_training = int(oracle.get_number_of_ratings() * percentage_of_ratings_for_training)
    assert number_of_ratings_for_training < oracle.get_number_of_ratings()

    training_data = []
    nb=0
    for i in range(number_of_ratings_for_training):
        nb=nb+1
        [user, item, rating] = oracle.get_random_exemple()
        while (user, item) in used:
            [user, item, rating] = oracle.get_random_exemple()
        assert (user, item) not in used
        used.add((user, item))

        # Add Gaussian noise to the rating
        noisy_rating = rating + np.random.normal(0, noise)

        noisy_rating = np.clip(noisy_rating, 1, 5)
        training_data.append((user, item, noisy_rating))

    algo.fit(training_data)


    M = Mean("Erreur au carré")
    while len(used) < oracle.get_number_of_ratings():
        [user, item, rating] = oracle.get_random_exemple()
        while (user, item) in used:
            [user, item, rating] = oracle.get_random_exemple()
        assert (user, item) not in used
        used.add((user, item))

        prediction = algo.estimate(user, item)

        # Calculate squared error between actual rating and prediction
        squared_error = (rating - prediction) ** 2
        M.add(squared_error)

        if M.nb % 1000 == 0:
            # M.print()
            if M.getMeanMax() - M.getMeanMin() < 0.01:
                break

    print(f"RMSE = {math.sqrt(M.getMean()):.2f} +- {math.sqrt(M.getMeanMax() - M.getMean()):.2f}")
    return math.sqrt(M.getMean()), math.sqrt(M.getMeanMax() - M.getMean())
