# -*- coding: utf-8 -*-
"""
Module: EvaluationData

Prepares various train/test splits and data structures for evaluating
Surprise recommendation algorithms, including full data evaluation,
accuracy testing, leave-one-out for top-N metrics, and similarity-based
diversity analysis.
"""
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline

class EvaluationData:
    """
    Encapsulates dataset splits and auxiliary data needed for algorithm evaluation.

    Attributes:
        rankings (dict):         Precomputed popularity rankings for items.
        fullTrainSet (Trainset): Complete training set built from the full dataset.
        fullAntiTestSet (list):  All user-item pairs not in the full training set.
        trainSet (Trainset):     Random 75% train split.
        testSet (list):          Remaining 25% test split.
        LOOCVTrain (Trainset):   Train set with one rating left out per user.
        LOOCVTest (list):        The single left-out rating per user.
        LOOCVAntiTestSet (list): All other user-item pairs for LOOCV train.
        simsAlgo (Algo):         Fitted KNNBaseline algorithm for computing item similarities.
    """
    def __init__(self, data, popularityRankings):
        """
        Initialize evaluation data by creating necessary splits and similarity model.

        Args:
            data (Dataset): A Surprise Dataset object.
            popularityRankings (dict[int, int]): MovieID -> popularity rank mapping.
        """
        # Store popularity ranks for novelty evaluation
        self.rankings = popularityRankings
        
        # 1) Full train set for global evaluations (coverage, diversity, novelty)
        self.fullTrainSet = data.build_full_trainset()
        self.fullAntiTestSet = self.fullTrainSet.build_anti_testset()
        
        # 2) 75/25 random split for accuracy metrics (RMSE, MAE)
        self.trainSet, self.testSet = train_test_split(data, test_size=.25, random_state=1)
        
        # 3) Leave-One-Out split for top-N metrics (HR, ARHR, etc.)
        # Build a "leave one out" train/test split for evaluating top-N recommenders
        # And build an anti-test-set for building predictions
        LOOCV = LeaveOneOut(n_splits=1, random_state=1)
        # Only one split needed; yields a trainset/testset pair
        for train, test in LOOCV.split(data):
            self.LOOCVTrain = train
            self.LOOCVTest = test
        # Anti-test for building all predictions in LOOCV setting
        self.LOOCVAntiTestSet = self.LOOCVTrain.build_anti_testset()
        
        # 4) Fit item-item similarity model on full data for diversity
        sim_options = {
            'name': 'cosine',   # Cosine similarity between item feature vectors
            'user_based': False # Compute item-item similarities
        }
        self.simsAlgo = KNNBaseline(sim_options=sim_options)
        self.simsAlgo.fit(self.fullTrainSet)
            
    def GetFullTrainSet(self):
        """
        Returns:
            Trainset: Full training set of all ratings.
        """
        return self.fullTrainSet
    
    def GetFullAntiTestSet(self):
        """
        Returns:
            list: All (user, item, global_mean) not in fullTrainSet.
        """
        return self.fullAntiTestSet
    
    def GetAntiTestSetForUser(self, testSubject):
        """
        Build anti-test set for a single user to find all unrated items.

        Args:
            testSubject (int): Raw user ID for which to generate anti-test pairs.

        Returns:
            list of tuples: (userID, itemID, fill_value) for all items the user hasn't rated.
        """
        trainset = self.fullTrainSet
        fill = trainset.global_mean
        anti_testset = []
        # Convert raw user ID to inner ID for lookup
        u = trainset.to_inner_uid(str(testSubject))
        # Items the user has already rated
        user_items = set([j for (j, _) in trainset.ur[u]])
        # For every item in the dataset, if not rated, add to anti-test
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                                 i in trainset.all_items() if
                                 i not in user_items]
        return anti_testset

    def GetTrainSet(self):
        """
        Returns:
            Trainset: 75% randomized training split.
        """
        return self.trainSet
    
    def GetTestSet(self):
        """
        Returns:
            list: 25% randomized test split.
        """
        return self.testSet
    
    def GetLOOCVTrainSet(self):
        """
        Returns:
            Trainset: Leave-One-Out train split.
        """
        return self.LOOCVTrain
    
    def GetLOOCVTestSet(self):
        """
        Returns:
            list: The single left-out ratings for each user.
        """
        return self.LOOCVTest
    
    def GetLOOCVAntiTestSet(self):
        """
        Returns:
            list: Anti-test set for LOOCV train set (all unseen pairs).
        """
        return self.LOOCVAntiTestSet
    
    def GetSimilarities(self):
        """
        Returns:
            Algo: Fitted similarity algorithm (KNNBaseline) on full data.
        """
        return self.simsAlgo
    
    def GetPopularityRankings(self):
        """
        Returns:
            dict: MovieID -> popularity rank mapping passed in constructor.
        """
        return self.rankings