# -*- coding: utf-8 -*-
"""
Custom SVD++ Algorithm Implementation without Surprise but compatible with Framework evaluation
"""
import sys
import os
# Get the absolute path to the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Libs used for PureSVDpp
from EvaluationFramework.MovieLens import MovieLens
import numpy as np
import math
import random
from collections import defaultdict

# Libs used for AdaptedSVDpp
from surprise import AlgoBase, PredictionImpossible

# Libs used for Evaluation
from utils.LoadMovieLensData import LoadMovieLensData
from EvaluationFramework.Evaluator import Evaluator
from surprise import NormalPredictor
import random


class PureSVDpp:
    """
    A pure Python implementation of SVD++ for matrix factorization
    
    SVD++ extends SVD by incorporating implicit feedback (which items users have rated)
    in addition to the explicit ratings.
    """
    
    def __init__(self, n_factors=100, n_epochs=20, lr=0.005, reg=0.02, implicit_weight=0.1):
        """
        Initialize SVD++ parameters
        
        Args:
            n_factors (int): Number of latent factors
            n_epochs (int): Number of iterations for SGD
            lr (float): Learning rate for SGD
            reg (float): Regularization term for SGD
            implicit_weight (float): Weight for implicit feedback factor
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr  # Learning rate
        self.reg = reg  # Regularization factor
        self.implicit_weight = implicit_weight  # Weight for implicit feedback
        
        # These will be initialized during fitting
        self.user_factors = None
        self.item_factors = None
        self.item_implicit_factors = None  # New in SVD++
        self.global_mean = 0
        self.user_biases = {}
        self.item_biases = {}
        
        # Mappings for user and item IDs
        self.user_id_to_index = {}
        self.item_id_to_index = {}
        self.user_index_to_id = {}
        self.item_index_to_id = {}
        
        # User-item interactions (implicit feedback)
        self.user_rated_items = defaultdict(list)  # Dictionary of user_id -> list of item indices
    
    def fit(self, ratings_data):
        """
        Train the SVD++ model using SGD
        
        Args:
            ratings_data: List of (user_id, item_id, rating) tuples
        """
        print("Processing ratings data for SVD++ matrix factorization...")
        
        # Extract all users, items, and ratings
        users = set()
        items = set()
        ratings_list = []
        
        for user_id, item_id, rating in ratings_data:
            users.add(user_id)
            items.add(item_id)
            ratings_list.append(rating)
        
        # Create mappings for users and items
        self.users = sorted(list(users))
        self.items = sorted(list(items))
        
        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(self.users)}
        self.item_id_to_index = {item_id: idx for idx, item_id in enumerate(self.items)}
        
        self.user_index_to_id = {idx: user_id for user_id, idx in self.user_id_to_index.items()}
        self.item_index_to_id = {idx: item_id for item_id, idx in self.item_id_to_index.items()}
        
        # Calculate global mean rating
        self.global_mean = sum(ratings_list) / len(ratings_list) if ratings_list else 0
        
        # Initialize parameters
        n_users = len(self.users)
        n_items = len(self.items)
        
        # User factors matrix (users × factors)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        
        # Item factors matrix (items × factors)
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # Item implicit factors matrix - new in SVD++ (items × factors)
        self.item_implicit_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # Initialize biases
        self.user_biases = {u: 0.0 for u in self.users}
        self.item_biases = {i: 0.0 for i in self.items}
        
        # Build user-item interaction data for implicit feedback
        print("Building implicit feedback data...")
        for user_id, item_id, _ in ratings_data:
            if user_id in self.user_id_to_index and item_id in self.item_id_to_index:
                user_idx = self.user_id_to_index[user_id]
                item_idx = self.item_id_to_index[item_id]
                self.user_rated_items[user_id].append(item_idx)
        
        # Pre-calculate the sqrt of user rated items lengths (for normalization)
        self.sqrt_user_rated_counts = {}
        for user_id, rated_items in self.user_rated_items.items():
            self.sqrt_user_rated_counts[user_id] = 1.0 / math.sqrt(len(rated_items)) if rated_items else 0
        
        # Train using SGD (stochastic gradient descent)
        print("Training SVD++ model using Stochastic Gradient Descent...")
        
        # Pre-create a shuffled list of ratings for each epoch
        shuffled_data = list(ratings_data)
        
        # Train over multiple epochs
        for epoch in range(self.n_epochs):
            # Shuffle the data each epoch
            random.shuffle(shuffled_data)
            
            squared_error = 0
            
            for user_id, item_id, rating in shuffled_data:
                # Skip if user or item not in our mappings
                if user_id not in self.user_id_to_index or item_id not in self.item_id_to_index:
                    continue
                    
                # Get indices
                user_idx = self.user_id_to_index[user_id]
                item_idx = self.item_id_to_index[item_id]
                
                # Get implicit feedback factor - sum of implicit item factors for items user has rated
                implicit_sum = np.zeros(self.n_factors)
                sqrt_count = self.sqrt_user_rated_counts[user_id]
                
                if sqrt_count > 0:
                    for implicit_item_idx in self.user_rated_items[user_id]:
                        implicit_sum += self.item_implicit_factors[implicit_item_idx]
                    
                    # Normalize by sqrt(number of items rated)
                    implicit_sum *= sqrt_count
                
                # Compute prediction with implicit feedback component
                # prediction = global_mean + user_bias + item_bias + 
                #              (user_factors + implicit_feedback)·item_factors
                pred = (
                    self.global_mean
                    + self.user_biases[user_id]
                    + self.item_biases[item_id]
                    + np.dot(
                        self.user_factors[user_idx] + self.implicit_weight * implicit_sum,
                        self.item_factors[item_idx]
                    )
                )
                
                # Calculate error
                error = rating - pred
                squared_error += error ** 2
                
                # Update biases
                self.user_biases[user_id] += self.lr * (error - self.reg * self.user_biases[user_id])
                self.item_biases[item_id] += self.lr * (error - self.reg * self.item_biases[item_id])
                
                # Save old factors for updates
                old_user_factors = self.user_factors[user_idx].copy()
                old_item_factors = self.item_factors[item_idx].copy()
                
                # Update user factors
                self.user_factors[user_idx] += self.lr * (
                    error * old_item_factors - self.reg * old_user_factors
                )
                
                # Update item factors
                self.item_factors[item_idx] += self.lr * (
                    error * (old_user_factors + self.implicit_weight * implicit_sum) 
                    - self.reg * old_item_factors
                )
                
                # Update implicit item factors (for all items user has rated)
                if sqrt_count > 0:
                    for implicit_item_idx in self.user_rated_items[user_id]:
                        self.item_implicit_factors[implicit_item_idx] += self.lr * (
                            error * self.implicit_weight * sqrt_count * old_item_factors
                            - self.reg * self.item_implicit_factors[implicit_item_idx]
                        )
            
            # Print epoch progress
            rmse = math.sqrt(squared_error / len(shuffled_data))
            print(f"Epoch {epoch+1}/{self.n_epochs}: RMSE = {rmse:.4f}")
            
            # Adaptive learning rate - reduce over time
            self.lr *= 0.9
        
        print("SVD++ training complete!")
        return self
    
    def predict(self, user_id, item_id):
        """
        Predict rating for a user-item pair
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating or None if prediction impossible
        """
        # Check if user and item exist in the training data
        if user_id not in self.user_id_to_index or item_id not in self.item_id_to_index:
            # For cold start, return global mean
            return self.global_mean
        
        # Get indices
        user_idx = self.user_id_to_index[user_id]
        item_idx = self.item_id_to_index[item_id]
        
        # Calculate implicit feedback component
        implicit_sum = np.zeros(self.n_factors)
        sqrt_count = self.sqrt_user_rated_counts.get(user_id, 0)
        
        if sqrt_count > 0:
            for implicit_item_idx in self.user_rated_items[user_id]:
                implicit_sum += self.item_implicit_factors[implicit_item_idx]
            
            # Normalize by sqrt(number of items rated)
            implicit_sum *= sqrt_count
        
        # Compute prediction with implicit feedback
        prediction = (
            self.global_mean
            + self.user_biases[user_id]
            + self.item_biases[item_id]
            + np.dot(
                self.user_factors[user_idx] + self.implicit_weight * implicit_sum,
                self.item_factors[item_idx]
            )
        )
        
        # Clip to valid rating range
        prediction = max(1.0, min(5.0, prediction))
        
        return prediction
    
    def get_user_factors(self, user_id):
        """Get latent factors for a specific user, including implicit feedback"""
        if user_id not in self.user_id_to_index:
            return None
            
        user_idx = self.user_id_to_index[user_id]
        
        # Calculate combined factors including implicit feedback
        base_factors = self.user_factors[user_idx]
        
        implicit_sum = np.zeros(self.n_factors)
        sqrt_count = self.sqrt_user_rated_counts.get(user_id, 0)
        
        if sqrt_count > 0:
            for implicit_item_idx in self.user_rated_items[user_id]:
                implicit_sum += self.item_implicit_factors[implicit_item_idx]
            
            implicit_sum *= sqrt_count
        
        # Return combined factors: explicit + implicit
        return base_factors + self.implicit_weight * implicit_sum
    
    def get_item_factors(self, item_id):
        """Get latent factors for a specific item"""
        if item_id in self.item_id_to_index:
            return self.item_factors[self.item_id_to_index[item_id]]
        return None


class AdaptedSVDpp(AlgoBase):
    """
    Adapter class that wraps PureSVDpp to make it compatible with Surprise's AlgoBase
    for evaluation with the Framework
    """
    
    def __init__(self, n_factors=100, n_epochs=20, lr=0.005, reg=0.02, implicit_weight=0.1):
        """Initialize SVD++ parameters"""
        AlgoBase.__init__(self)
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.implicit_weight = implicit_weight
        self.pure_svdpp = PureSVDpp(
            n_factors=n_factors, 
            n_epochs=n_epochs, 
            lr=lr, 
            reg=reg, 
            implicit_weight=implicit_weight
        )
        
    def fit(self, trainset):
        """
        Fit the algorithm to the trainset
        
        Args:
            trainset: A surprise trainset
            
        Returns:
            self
        """
        AlgoBase.fit(self, trainset)
        
        # Convert Surprise trainset to our format
        ratings_data = []
        for u, i, r in trainset.all_ratings():
            # Convert internal IDs to raw IDs
            user_id = trainset.to_raw_uid(u)
            item_id = int(trainset.to_raw_iid(i))
            rating = r
            
            ratings_data.append((user_id, item_id, rating))
        
        # Train our pure Python SVD++ implementation
        self.pure_svdpp.fit(ratings_data)
        
        # Store trainset for later use
        self.trainset = trainset
        
        return self
    
    def estimate(self, u, i):
        """
        Estimate rating for a user-item pair
        
        Args:
            u: Internal user ID
            i: Internal item ID
            
        Returns:
            Predicted rating
        """
        try:
            # Try to convert internal IDs to raw IDs
            user_id = self.trainset.to_raw_uid(u)
            item_id = int(self.trainset.to_raw_iid(i))
            
            # Get prediction from pure SVD++
            prediction = self.pure_svdpp.predict(user_id, item_id)
            
            if prediction is None:
                raise PredictionImpossible("Cannot make prediction for this user-item pair")
                
            return prediction
        
        except ValueError:
            # Handle unknown items or users
            raise PredictionImpossible(f"User or item is unknown: {u}, {i}")


# For testing and evaluation
if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    
    # Load data
    (ml, evaluationData, rankings) = LoadMovieLensData()
    
    # Create evaluator
    evaluator = Evaluator(evaluationData, rankings)
    
    # Add algorithms
    svdpp_recommender = AdaptedSVDpp(n_factors=50, n_epochs=20, lr=0.005, reg=0.02, implicit_weight=0.1)
    evaluator.AddAlgorithm(svdpp_recommender, "SVD++")
    
    random_predictor = NormalPredictor()
    evaluator.AddAlgorithm(random_predictor, "Random")
    
    # Run evaluation
    print("Evaluating algorithms...")
    evaluator.Evaluate(False)
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)