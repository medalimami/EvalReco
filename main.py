from evaluate import evaluate
from plot_results import plot_results

# Import the Oracles
from Oracle.OracleNMF import *
from Oracle.OracleSurpriseDataset import *
from Oracle.OracleRT import *

# Import the Algorithms
from Algo.SVDpp import AlgoSVDpp
from Algo.SVD import AlgoSVD
from Algo.NMF import AlgoNMF
from Algo.Base import AlgoBase
from Algo.KNN import AlgoKNN

# Import the Bagging methods
from Algo.bagging import AlgoBagging
from Algo.baggingv2 import AlgoBaggingV2

# Set the number of users/items
num_users = 500
num_items = 200


# Generate the oracles
oracle = OracleNMF(num_users=num_users, num_items=num_items, rank=90, max_rating=5)
#genres_list = ["Action", "Comedy", "Drama", "Adventure", "Sci-Fi"]
#oracle = OracleRandomTrees(num_users=num_users, num_items=num_items, genres_list=genres_list, leaf_probability=0.5)
#oracle = OracleSurpriseDataset("ml-100k")
#oracle = OracleSurpriseDataset("ml-1m")


# Add the algorithms to test
algobase = AlgoBase(rating_scale=(1, 5))
algosvd = AlgoSVD(rating_scale=(1, 5))
algosvdpp = AlgoSVDpp(rating_scale=(1, 5))
algonmf = AlgoNMF(rating_scale=(1, 5))
algoknn = AlgoKNN(rating_scale=(1, 5))


# Set the parameters
percentage_of_ratings_for_training = [0.9,0.5,0.1,0.01]
noise_levels = [0,1,2]
algos = [AlgoBagging(base_model=algosvdpp, num_models=21),AlgoBagging(base_model=algosvd, num_models=21),
         AlgoBagging(base_model=algoknn, num_models=21), AlgoBagging(base_model=algobase, num_models=21),
         AlgoBagging(base_model=algonmf, num_models=21)]

# Add a variabel to store the results to display them in the curves
results = []

algorithms = []
# Calculate the RMSE for each model and oracle parameter
for algo in algos:
  algo_name = algo.__class__.__name__
  base_model_name = algo.get_base_model_name()
  algorithms.append(f"{base_model_name} ({algo_name})")
  for training_percentage in percentage_of_ratings_for_training:
    for noise_level in noise_levels:
      print(f"Model : {algo_name} ({base_model_name}) | Percentage of training ratings : {training_percentage*100}% | Noise Level : {noise_level}")
      rmse,error = evaluate(algo=algo, oracle=oracle, percentage_of_ratings_for_training=training_percentage,noise=noise_level)
      number_of_ratings_for_training = int(oracle.get_number_of_ratings() * training_percentage)
      print("*"*90)
      results.append({
        'model': f"{base_model_name} ({algo_name})",
        'num_ratings': number_of_ratings_for_training,
        'noise_level': noise_level,
        'rmse': f'{rmse:.2f} - {error:.2f}'
      })

plot_results(results,noise_levels,percentage_of_ratings_for_training,algorithms,oracle)