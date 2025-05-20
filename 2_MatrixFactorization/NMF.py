# -*- coding: utf-8 -*-
"""
Custom Non-negative Matrix Factorization (NMF) Algorithm Implementation 
without Surprise but compatible with Framework evaluation
"""
import sys
import os
# Get the absolute path to the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Libs used for PureNMF
from EvaluationFramework.MovieLens import MovieLens
import numpy as np
import math
import random

# Libs used for AdaptedNMF
from surprise import AlgoBase, PredictionImpossible

# Libs used for Evaluation
from utils.LoadMovieLensData import LoadMovieLensData
from EvaluationFramework.Evaluator import Evaluator
from surprise import NormalPredictor
import random


class PureNMF:
    """
    A pure Python implementation of Non-negative Matrix Factorization (NMF)
    
    NMF decomposes the ratings matrix into two non-negative matrices:
    R ≈ P·Q^T, where P and Q are non-negative matrices.
    """
    
    def __init__(self, n_factors=15, n_epochs=50, lr=0.01, reg=0.02, beta=0.02):
        """
        Initialize NMF parameters
        
        Args:
            n_factors (int): Number of latent factors
            n_epochs (int): Number of iterations for SGD
            lr (float): Learning rate for SGD
            reg (float): Regularization term for SGD
            beta (float): Regularization for non-negativity constraint
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr  # Learning rate
        self.reg = reg  # Regularization factor
        self.beta = beta  # Non-negativity regularization
        
        # These will be initialized during fitting
        self.user_factors = None
        self.item_factors = None
        self.global_mean = 0
        
        # Mappings for user and item IDs
        self.user_id_to_index = {}
        self.item_id_to_index = {}
        self.user_index_to_id = {}
        self.item_index_to_id = {}
    
    def fit(self, ratings_data):
        """
        Train the NMF model using gradient descent
        
        Args:
            ratings_data: List of (user_id, item_id, rating) tuples
        """
        print("Processing ratings data for NMF matrix factorization...")
        
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
        
        # Initialize factor matrices with small positive random values
        # This ensures they start as non-negative
        self.user_factors = np.random.uniform(0.01, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.uniform(0.01, 0.1, (n_items, self.n_factors))
        
        # Create mapping of (user, item) -> rating for faster lookups
        ratings_dict = {(user_id, item_id): rating for user_id, item_id, rating in ratings_data}
        
        # Normalize ratings to [0, 1] for better NMF performance
        min_rating = min(ratings_list)
        max_rating = max(ratings_list)
        rating_range = max_rating - min_rating
        
        if rating_range > 0:
            normalized_ratings = {k: (v - min_rating) / rating_range for k, v in ratings_dict.items()}
        else:
            normalized_ratings = ratings_dict
            
        self.min_rating = min_rating
        self.rating_range = rating_range
        
        # Train using gradient descent
        print("Training NMF model using gradient descent...")
        
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
                
                # Get normalized rating
                norm_rating = normalized_ratings[(user_id, item_id)]
                
                # Compute prediction
                pred = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
                
                # Calculate error (normalized)
                error = norm_rating - pred
                squared_error += error ** 2
                
                # Save old factors for updates
                old_user_factors = self.user_factors[user_idx].copy()
                old_item_factors = self.item_factors[item_idx].copy()
                
                # Update user factors with gradient descent
                # Include non-negativity constraint through regularization
                for f in range(self.n_factors):
                    update = (
                        self.lr * (error * old_item_factors[f] - self.reg * old_user_factors[f])
                    )
                    
                    # Ensure non-negativity with soft constraint
                    if old_user_factors[f] + update < 0:
                        update = -old_user_factors[f] * self.beta  # Reduce negative updates
                        
                    self.user_factors[user_idx, f] += update
                    
                    # Ensure strict non-negativity
                    self.user_factors[user_idx, f] = max(0, self.user_factors[user_idx, f])
                
                # Update item factors with gradient descent
                # Include non-negativity constraint
                for f in range(self.n_factors):
                    update = (
                        self.lr * (error * old_user_factors[f] - self.reg * old_item_factors[f])
                    )
                    
                    # Ensure non-negativity with soft constraint
                    if old_item_factors[f] + update < 0:
                        update = -old_item_factors[f] * self.beta  # Reduce negative updates
                        
                    self.item_factors[item_idx, f] += update
                    
                    # Ensure strict non-negativity
                    self.item_factors[item_idx, f] = max(0, self.item_factors[item_idx, f])
            
            # Print epoch progress
            rmse = math.sqrt(squared_error / len(shuffled_data))
            print(f"Epoch {epoch+1}/{self.n_epochs}: RMSE = {rmse:.4f}")
            
            # Adaptive learning rate - reduce over time
            self.lr *= 0.95
        
        print("NMF training complete!")
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
        
        # Compute normalized prediction
        norm_pred = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        # Clip normalized prediction to [0, 1]
        norm_pred = max(0, min(1, norm_pred))
        
        # Convert back to original rating scale
        prediction = norm_pred * self.rating_range + self.min_rating
        
        # Clip to valid rating range (typically 1-5)
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


class AdaptedNMF(AlgoBase):
    """
    Adapter class that wraps PureNMF to make it compatible with Surprise's AlgoBase
    for evaluation with the Framework
    """
    
    def __init__(self, n_factors=15, n_epochs=50, lr=0.01, reg=0.02, beta=0.02):
        """Initialize NMF parameters"""
        AlgoBase.__init__(self)
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.beta = beta
        self.pure_nmf = PureNMF(
            n_factors=n_factors, 
            n_epochs=n_epochs, 
            lr=lr, 
            reg=reg,
            beta=beta
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
        
        # Train our pure Python NMF implementation
        self.pure_nmf.fit(ratings_data)
        
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
            
            # Get prediction from pure NMF
            prediction = self.pure_nmf.predict(user_id, item_id)
            
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
    nmf_recommender = AdaptedNMF(n_factors=15, n_epochs=50, lr=0.01, reg=0.02, beta=0.02)
    evaluator.AddAlgorithm(nmf_recommender, "NMF")
    
    random_predictor = NormalPredictor()
    evaluator.AddAlgorithm(random_predictor, "Random")
    
    # Run evaluation
    print("Evaluating algorithms...")
    evaluator.Evaluate(False)
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)