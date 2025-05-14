# -*- coding: utf-8 -*-
"""
Module: EvaluatedAlgorithm

Defines a wrapper for Surprise recommendation algorithms that runs a suite of
evaluation tests including accuracy (RMSE, MAE), top-N recommendation metrics,
coverage, diversity, and novelty.
"""
from RecommenderMetrics import RecommenderMetrics
from EvaluationData import EvaluationData

class EvaluatedAlgorithm:
    """
    Encapsulates a recommendation algorithm and provides methods to evaluate it
    against various metrics using train/test splits and leave-one-out testing.

    Attributes:
        algorithm: A Surprise algorithm instance (must implement .fit and .test).
        name (str): A descriptive name for the algorithm (used in reporting).
    """

    def __init__(self, algorithm, name):
        """
        Initialize with a preconfigured Surprise algorithm.

        Args:
            algorithm: An instance of a Surprise recommender (e.g., SVD, KNN).
            name (str): Label for this algorithm instance.
        """
        self.algorithm = algorithm
        self.name = name
        
    def Evaluate(self, evaluationData, doTopN, n=10, verbose=True):
        """
        Run full evaluation pipeline on the provided data.

        Steps:
            1. Fit on train/test split and compute RMSE, MAE.
            2. If doTopN is True:
               a) Leave-one-out test: fit on LOOCV train, predict left-out and anti-test.
                  - Compute hit rate (HR), cumulative hit rate (cHR), and average
                    reciprocal hit rank (ARHR) on left-out data.
               b) Full data test: fit on all training data, predict anti-test.
                  - Compute coverage, diversity, and novelty of top-N recs.

        Args:
            evaluationData (EvaluationData): Object providing train/test splits
                                             and required data structures.
            doTopN (bool): Whether to evaluate top-N metrics.
            n (int): Number of top recommendations per user for top-N tests.
            verbose (bool): Whether to print progress messages.

        Returns:
            dict: Mapping metric names (e.g., 'RMSE', 'HR', 'Diversity') to values.
        """
        metrics = {}
        # 1: Accuracy evaluation on standard train/test split
        if (verbose):
            print("Evaluating accuracy...")
        self.algorithm.fit(evaluationData.GetTrainSet())
        predictions = self.algorithm.test(evaluationData.GetTestSet())
        metrics["RMSE"] = RecommenderMetrics.RMSE(predictions)
        metrics["MAE"] = RecommenderMetrics.MAE(predictions)
        
        if (doTopN):
            # 2a: Leave-one-out evaluation for hit/rank metrics
            if (verbose):
                print("Evaluating top-N with leave-one-out...")
            self.algorithm.fit(evaluationData.GetLOOCVTrainSet())
            leftOutPredictions = self.algorithm.test(evaluationData.GetLOOCVTestSet())        
            # Build predictions for all ratings not in the training set
            allPredictions = self.algorithm.test(evaluationData.GetLOOCVAntiTestSet())
            # Build top-N recommendations for each user
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)
            
            if (verbose):
                print("Computing hit-rate and rank metrics...")
            # See how often we recommended a movie the user actually rated
            metrics["HR"] = RecommenderMetrics.HitRate(topNPredicted, leftOutPredictions)   
            # See how often we recommended a movie the user actually liked
            metrics["cHR"] = RecommenderMetrics.CumulativeHitRate(topNPredicted, leftOutPredictions)
            # Compute ARHR
            metrics["ARHR"] = RecommenderMetrics.AverageReciprocalHitRank(topNPredicted, leftOutPredictions)
        
            # 2b: Full data coverage, diversity, novelty
            #Evaluate properties of recommendations on full training set
            if (verbose):
                print("Computing recommendations with full data set...")
            self.algorithm.fit(evaluationData.GetFullTrainSet())
            allPredictions = self.algorithm.test(evaluationData.GetFullAntiTestSet())
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)
           
            if (verbose):
                print("Analyzing coverage, diversity, and novelty...")
            # Coverage: fraction of users with >=1 recommendation above threshold
            # Print user coverage with a minimum predicted rating of 4.0:
            metrics["Coverage"] = RecommenderMetrics.UserCoverage(  topNPredicted, 
                                                                   evaluationData.GetFullTrainSet().n_users, 
                                                                   ratingThreshold=4.0)
            # Diversity: 1 - average pairwise similarity of recommendations
            metrics["Diversity"] = RecommenderMetrics.Diversity(topNPredicted, evaluationData.GetSimilarities())
            
            # Novelty: average popularity rank of recommended items
            metrics["Novelty"] = RecommenderMetrics.Novelty(topNPredicted, 
                                                            evaluationData.GetPopularityRankings())
        
        if (verbose):
            print("Analysis complete.")
    
        return metrics
    
    def GetName(self):
        """
        Get the name label of this evaluated algorithm.

        Returns:
            str: The algorithm's name.
        """
        return self.name
    
    def GetAlgorithm(self):
        """
        Access the underlying Surprise algorithm instance.

        Returns:
            Surprise algorithm object.
        """
        return self.algorithm