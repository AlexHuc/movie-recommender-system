# -*- coding: utf-8 -*-
"""
Module: Evaluator

Provides a convenient interface to evaluate multiple recommendation algorithms
on a dataset with various metrics, and to sample top-N recommendations for a user.
"""
from EvaluationFramework.EvaluationData import EvaluationData
from EvaluationFramework.EvaluatedAlgorithm import EvaluatedAlgorithm

class Evaluator:
    """
    Orchestrates the evaluation of recommender algorithms.

    Attributes:
        algorithms (list): List of EvaluatedAlgorithm instances to be evaluated.
        dataset (EvaluationData): Prepared data splits and structures for evaluation.
    """
    algorithms = []
    
    def __init__(self, dataset, rankings):
        """
        Initialize Evaluator with dataset and popularity rankings.

        Args:
            dataset: A Surprise Dataset object containing the ratings data.
            rankings (dict): MovieID -> popularity rank mapping for novelty metric.
        """
        # Prepare evaluation data splits and similarity model
        ed = EvaluationData(dataset, rankings)
        self.dataset = ed
        
    def AddAlgorithm(self, algorithm, name):
        """
        Add a recommendation algorithm to the evaluation list.

        Args:
            algorithm: A configured Surprise algorithm instance.
            name (str): Descriptive name for this algorithm.
        """
        alg = EvaluatedAlgorithm(algorithm, name)
        self.algorithms.append(alg)
        
    def Evaluate(self, doTopN):
        """
        Execute evaluation on all added algorithms.

        Computes accuracy metrics (RMSE, MAE) on a train/test split.
        If doTopN is True, also evaluates top-N metrics: hit rate (HR),
        cumulative hit rate (cHR), average reciprocal hit rank (ARHR),
        coverage, diversity, and novelty.

        Args:
            doTopN (bool): Whether to perform top-N recommendation evaluation.

        Returns:
            dict: Mapping algorithm names to their metrics dict.
        """
        results = {}
        # Run each algorithm's Evaluate and collect metrics
        for algorithm in self.algorithms:
            print("Evaluating ", algorithm.GetName(), "...")
            results[algorithm.GetName()] = algorithm.Evaluate(self.dataset, doTopN)

        # Print formatted results
        print("\n")
        if (doTopN):
            # Header for top-N metrics
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                    "Algorithm", "RMSE", "MAE", "HR", "cHR", "ARHR", "Coverage", "Diversity", "Novelty"))
            # Print each algorithm's metrics
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                        name, metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["cHR"], metrics["ARHR"],
                                      metrics["Coverage"], metrics["Diversity"], metrics["Novelty"]))
        else:
            # Header for accuracy-only metrics
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"]))

        # Legend explaining metrics  
        print("\nLegend:\n")
        print("RMSE:      Root Mean Squared Error. Lower values mean better accuracy.")
        print("MAE:       Mean Absolute Error. Lower values mean better accuracy.")
        if (doTopN):
            print("HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better.")
            print("cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better.")
            print("ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better." )
            print("Coverage:  Ratio of users for whom recommendations above a certain threshold exist. Higher is better.")
            print("Diversity: 1-S, where S is the average similarity score between every possible pair of recommendations")
            print("           for a given user. Higher means more diverse.")
            print("Novelty:   Average popularity rank of recommended items. Higher means more novel.")
        
    def SampleTopNRecs(self, ml, testSubject=85, k=10):
        """
        Generate and print top-N recommendations for a specific user using each algorithm.

        Args:
            ml: A MovieLens helper instance with getMovieName method.
            testSubject (int): Raw user ID for whom to sample recommendations.
            k (int): Number of recommendations to display.
        """
        for algo in self.algorithms:
            print("\nUsing recommender ", algo.GetName())
            
            # Fit on full training set before sampling
            print("\nBuilding recommendation model...")
            trainSet = self.dataset.GetFullTrainSet()
            algo.GetAlgorithm().fit(trainSet)
            
            # Build anti-test set for this user to predict unrated items
            print("Computing recommendations...")
            testSet = self.dataset.GetAntiTestSetForUser(testSubject)
            predictions = algo.GetAlgorithm().test(testSet)
            
            # Collect and sort recommendations by estimated rating
            recommendations = []
            print ("\nWe recommend:")
            for userID, movieID, actualRating, estimatedRating, _ in predictions:
                intMovieID = int(movieID)
                recommendations.append((intMovieID, estimatedRating))
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            # Display top-k movie titles and scores
            for ratings in recommendations[:10]:
                print(ml.getMovieName(ratings[0]), ratings[1])
                

            
            
    
    