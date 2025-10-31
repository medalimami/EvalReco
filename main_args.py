import argparse
import pandas as pd
import os

from evaluate import evaluate
from plot_results import plot_results

# Import the oracles
from Oracle.OracleNMF import *
from Oracle.OracleSurpriseDataset import *
from Oracle.OracleRT import *

# Import the Algorithms
from Algo.SVDpp import AlgoSVDpp
from Algo.SVD import AlgoSVD
from Algo.NMF import AlgoNMF
from Algo.Base import AlgoBase
from Algo.KNN import AlgoKNN

# Import bagging methods
from Algo.bagging import AlgoBagging
from Algo.baggingv2 import AlgoBaggingV2



# Get the Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for evaluating recommendation algorithms.')
    parser.add_argument('--oracle', type=str, choices=['nmf', 'random_trees', 'ml-100k', 'ml-1m'], default='ml-100k', help='Type of oracle')
    parser.add_argument('--algorithm', nargs='+', type=str, choices=['svd', 'bagging_svd', 'baggingv2_svd', 'svdpp', 'bagging_svdpp', 'baggingv2_svdpp', 'nmf',
                                                                     'bagging_nmf', 'baggingv2_nmf', 'knn', 'bagging_knn', 'baggingv2_knn',
                                                                     'bagging_base', 'baggingv2_base', 'base'], default='svdpp', help='Type of recommendation algorithm')
    parser.add_argument('--rank', nargs='+', type=int, default=[], help='Rank for NMF oracle')
    parser.add_argument('--leaf_probability', nargs='+', type=float, default=[], help='Leaf probability for Random Trees oracle')
    parser.add_argument('--num_users', type=int, default=5, help='Number of users for Oracle')
    parser.add_argument('--num_items', type=int, default=5, help='Number of items for Oracle')
    parser.add_argument('--noise_levels', nargs='+', type=int, default=[0, 1, 2], help='List of noise levels for Oracle')
    parser.add_argument('--training_percentages', nargs='+', type=float, default=[90,50,10], help='List of training percentages for Oracle')
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()
    if not os.path.exists("results"):
        os.makedirs("results")
    # Create the algorithms based on the passed arguments
    algos = []
    if 'svd' in args.algorithm:
        algos.append(AlgoSVD(rating_scale=(1, 5)))
    if 'bagging_svd' in args.algorithm:
        algos.append(AlgoBagging(base_model=AlgoSVD(rating_scale=(1, 5)), num_models=21))
    if 'baggingv2_svd' in args.algorithm:
        algos.append(AlgoBaggingV2(base_model=AlgoSVD(rating_scale=(1, 5)), num_models=21, epsilon=0.1))

    if 'svdpp' in args.algorithm:
        algos.append(AlgoSVDpp(rating_scale=(1, 5)))
    if 'bagging_svdpp' in args.algorithm:
        algos.append(AlgoBagging(base_model=AlgoSVDpp(rating_scale=(1, 5)), num_models=21))
    if 'baggingv2_svdpp' in args.algorithm:
        algos.append(AlgoBaggingV2(base_model=AlgoSVDpp(rating_scale=(1, 5)), num_models=21, epsilon=0.1))

        # ajout baggingv2
    if 'nmf' in args.algorithm:
        algos.append(AlgoNMF(rating_scale=(1, 5)))
    if 'bagging_nmf' in args.algorithm:
        algos.append(AlgoBagging(base_model=AlgoNMF(rating_scale=(1, 5)), num_models=21))
    if 'baggingv2_nmf' in args.algorithm:
        algos.append(AlgoBaggingV2(base_model=AlgoNMF(rating_scale=(1, 5)), num_models=21, epsilon=0.1))


    if 'knn' in args.algorithm:
        algos.append(AlgoKNN(rating_scale=(1,5)))
    if 'bagging_knn' in args.algorithm:
        algos.append(AlgoBagging(base_model=AlgoKNN(rating_scale=(1, 5)), num_models=21))
    if 'baggingv2_knn' in args.algorithm:
        algos.append(AlgoBaggingV2(base_model=AlgoKNN(rating_scale=(1, 5)), num_models=21, epsilon=0.1))


    if 'base' in args.algorithm:
        algos.append(AlgoBase(rating_scale=(1, 5)))
    if 'bagging_base' in args.algorithm:
        algos.append(AlgoBagging(base_model=AlgoBase(rating_scale=(1, 5)), num_models=21))
    if 'baggingv2_base' in args.algorithm:
        algos.append(AlgoBaggingV2(base_model=AlgoBase(rating_scale=(1, 5)), num_models=21, epsilon=0.1))


    # Create an empty results variable de store the results
    results = []
    for param in args.rank or args.leaf_probability or [None]:
        if 'nmf' in args.oracle:
            oracle = OracleNMF(num_users=args.num_users, num_items=args.num_items, rank=param, max_rating=5)
        elif 'random_trees' in args.oracle:
            genres_list = ["Action", "Comedy", "Drama", "Adventure", "Sci-Fi"]
            param=param/100
            oracle = OracleRandomTrees(num_users=args.num_users, num_items=args.num_items, genres_list=genres_list, leaf_probability=param)
        elif 'ml-100k' in args.oracle:
            oracle = OracleSurpriseDataset("ml-100k")
        elif 'ml-1m' in args.oracle:
            oracle = OracleSurpriseDataset("ml-1m")

        for algo in algos:
            base_model_name = algo.get_base_model_name()
            algo_name = algo.__class__.__name__ if algo.__class__.__name__ != base_model_name else "Default"
            for training_ratio in args.training_percentages:
                training_ratio = training_ratio/100
                for noise_level in args.noise_levels:
                    rmse, error = evaluate(algo=algo, oracle=oracle,
                                           percentage_of_ratings_for_training=training_ratio, noise=noise_level)
                    number_of_ratings_for_training = int(oracle.get_number_of_ratings() * training_ratio)
                    results.append({
                        'model': f"{base_model_name} ({algo_name})",
                        'training_ratio': training_ratio,
                        'noise_level': noise_level,
                        'rank' if 'nmf' in args.oracle else 'leaf_probability': param,
                        'rmse-error': f'{rmse:.2f} - {error:.2f}'
                    })

    df_results = pd.DataFrame(results)
    df_results.to_csv(f'results/{args.algorithm}-{args.oracle}.csv')
    print(df_results)


    plot_results(df_results)