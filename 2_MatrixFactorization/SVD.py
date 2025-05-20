# -*- coding: utf-8 -*-
"""
Custom Singular Value Decomposition (SVD) Algorithm Implementation 
without Surprise but compatible with Framework evaluation
"""
import sys
import os
# Get the absolute path to the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Libs used for PureSVD
from EvaluationFramework.MovieLens import MovieLens
import numpy as np
import math
import random

# Libs used for AdaptedSVD
from surprise import AlgoBase, PredictionImpossible

# Libs used for Evaluation
from utils.LoadMovieLensData import LoadMovieLensData
from EvaluationFramework.Evaluator import Evaluator
from surprise import NormalPredictor
import random


class PureSVD:
    """
    A pure Python implementation of Singular Value Decomposition for matrix factorization
    """
    
    def __init__(self, n_factors=100, n_epochs=20, lr=0.005, reg=0.02):
        """
        Initialize SVD parameters
        
        Args:
            n_factors (int): Number of latent factors
            n_epochs (int): Number of iterations for SGD
            lr (float): Learning rate for SGD
            reg (float): Regularization term for SGD
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr  # Learning rate
        self.reg = reg  # Regularization factor
        
        # These will be initialized during fitting
        self.user_factors = None
        self.item_factors = None
        self.global_mean = 0
        self.user_biases = {}
        self.item_biases = {}
        
        # Mappings for user and item IDs
        self.user_id_to_index = {}
        self.item_id_to_index = {}
        self.user_index_to_id = {}
        self.item_index_to_id = {}
    
    def fit(self, ratings_data):
        """
        Train the SVD model using SGD
        
        Args:
            ratings_data: List of (user_id, item_id, rating) tuples
        """
        print("Processing ratings data for SVD matrix factorization...")
        
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
        
        # Initialize biases
        self.user_biases = {u: 0.0 for u in self.users}
        self.item_biases = {i: 0.0 for i in self.items}
        
        # Train using SGD (stochastic gradient descent)
        print("Training SVD model using Stochastic Gradient Descent...")
        
        # Keep track of the count of instances for each user and item for bias normalization
        user_counts = {u: 0 for u in self.users}
        item_counts = {i: 0 for i in self.items}
        
        for user_id, item_id, rating in ratings_data:
            user_counts[user_id] += 1
            item_counts[item_id] += 1
        
        # Pre-create a shuffled list of ratings for each epoch
        shuffled_data = list(ratings_data)
        
        # Train over multiple epochs
        for epoch in range(self.n_epochs):
            # Shuffle the data each epoch
            random.shuffle(shuffled_data)
            
            squared_error = 0
            
            for user_id, item_id, rating in shuffled_data:
                # Get indices
                user_idx = self.user_id_to_index[user_id]
                item_idx = self.item_id_to_index[item_id]
                
                # Compute prediction
                # prediction = global_mean + user_bias + item_bias + user_factors·item_factors
                pred = (
                    self.global_mean
                    + self.user_biases[user_id]
                    + self.item_biases[item_id]
                    + np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
                )
                
                # Calculate error
                error = rating - pred
                squared_error += error ** 2
                
                # Update biases
                self.user_biases[user_id] += self.lr * (error - self.reg * self.user_biases[user_id])
                self.item_biases[item_id] += self.lr * (error - self.reg * self.item_biases[item_id])
                
                # Update user and item factors
                user_factors_copy = self.user_factors[user_idx].copy()
                
                # Update user factors
                self.user_factors[user_idx] += self.lr * (error * self.item_factors[item_idx] - 
                                                         self.reg * self.user_factors[user_idx])
                
                # Update item factors
                self.item_factors[item_idx] += self.lr * (error * user_factors_copy -
                                                         self.reg * self.item_factors[item_idx])
            
            # Print epoch progress
            rmse = math.sqrt(squared_error / len(shuffled_data))
            print(f"Epoch {epoch+1}/{self.n_epochs}: RMSE = {rmse:.4f}")
            
            # Adaptive learning rate - reduce over time
            self.lr *= 0.9
        
        print("SVD training complete!")
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
        
        # Compute prediction
        prediction = (
            self.global_mean
            + self.user_biases[user_id]
            + self.item_biases[item_id]
            + np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        )
        
        # Clip to valid rating range
        prediction = max(1.0, min(5.0, prediction))
        
        return prediction
    
    def get_user_factors(self, user_id):
        """Get latent factors for a specific user"""
        if user_id in self.user_id_to_index:
            return self.user_factors[self.user_id_to_index[user_id]]
        return None
    
    def get_item_factors(self, item_id):
        """Get latent factors for a specific item"""
        if item_id in self.item_id_to_index:
            return self.item_factors[self.item_id_to_index[item_id]]
        return None


class AdaptedSVD(AlgoBase):
    """
    Adapter class that wraps PureSVD to make it compatible with Surprise's AlgoBase
    for evaluation with the Framework
    """
    
    def __init__(self, n_factors=100, n_epochs=20, lr=0.005, reg=0.02):
        """Initialize SVD parameters"""
        AlgoBase.__init__(self)
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.pure_svd = PureSVD(n_factors=n_factors, n_epochs=n_epochs, lr=lr, reg=reg)
        
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
        
        # Train our pure Python SVD implementation
        self.pure_svd.fit(ratings_data)
        
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
            
            # Get prediction from pure SVD
            prediction = self.pure_svd.predict(user_id, item_id)
            
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
    svd_recommender = AdaptedSVD(n_factors=50, n_epochs=20, lr=0.005, reg=0.02)
    evaluator.AddAlgorithm(svd_recommender, "SVD")
    
    # # Add another SVD with different parameters for comparison
    # svd_recommender2 = AdaptedSVD(n_factors=100, n_epochs=20, lr=0.01, reg=0.01)
    # evaluator.AddAlgorithm(svd_recommender2, "SVD-Large")
    
    random_predictor = NormalPredictor()
    evaluator.AddAlgorithm(random_predictor, "Random")
    
    # Run evaluation
    print("Evaluating algorithms...")
    evaluator.Evaluate(False)
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)