# -*- coding: utf-8 -*-
"""
Custom Probabilistic Matrix Factorization (PMF) Algorithm Implementation 
without Surprise but compatible with Framework evaluation
"""
import sys
import os
# Get the absolute path to the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Libs used for PurePMF
from EvaluationFramework.MovieLens import MovieLens
import numpy as np
import math
import random

# Libs used for AdaptedPMF
from surprise import AlgoBase, PredictionImpossible

# Libs used for Evaluation
from utils.LoadMovieLensData import LoadMovieLensData
from EvaluationFramework.Evaluator import Evaluator
from surprise import NormalPredictor
import random


class PurePMF:
    """
    A pure Python implementation of Probabilistic Matrix Factorization (PMF)
    
    PMF models the ratings as a probabilistic process with Gaussian noise:
    R = U·V^T + ε, where ε is Gaussian noise
    and places Gaussian priors on the latent factors U and V.
    """
    
    def __init__(self, n_factors=10, n_epochs=50, lr=0.005, user_reg=0.1, item_reg=0.1, 
                 lr_decay=0.95, min_lr=0.001, init_std=0.1):
        """
        Initialize PMF parameters
        
        Args:
            n_factors (int): Number of latent factors
            n_epochs (int): Number of iterations for SGD
            lr (float): Initial learning rate for SGD
            user_reg (float): Regularization for user factors
            item_reg (float): Regularization for item factors
            lr_decay (float): Learning rate decay per epoch
            min_lr (float): Minimum learning rate
            init_std (float): Standard deviation for initial factor values
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr  # Learning rate
        self.user_reg = user_reg  # User regularization factor
        self.item_reg = item_reg  # Item regularization factor
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.init_std = init_std
        
        # These will be initialized during fitting
        self.user_factors = None
        self.item_factors = None
        self.global_mean = 0
        
        # Mappings for user and item IDs
        self.user_id_to_index = {}
        self.item_id_to_index = {}
        self.user_index_to_id = {}
        self.item_index_to_id = {}
        
        # Rating normalization parameters
        self.min_rating = 1.0
        self.max_rating = 5.0
        self.rating_range = 4.0  # Default assuming 1-5 scale
    
    def fit(self, ratings_data):
        """
        Train the PMF model using SGD
        
        Args:
            ratings_data: List of (user_id, item_id, rating) tuples
        """
        print("Processing ratings data for PMF matrix factorization...")
        
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
        
        # Calculate global mean and rating bounds
        self.global_mean = sum(ratings_list) / len(ratings_list) if ratings_list else 0
        self.min_rating = min(ratings_list)
        self.max_rating = max(ratings_list)
        self.rating_range = self.max_rating - self.min_rating
        
        # Initialize parameters
        n_users = len(self.users)
        n_items = len(self.items)
        
        # Initialize factor matrices with random values from a Gaussian distribution
        # This follows the probabilistic nature of PMF
        self.user_factors = np.random.normal(0, self.init_std, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, self.init_std, (n_items, self.n_factors))
        
        # Normalize ratings to [0, 1] for better numerical stability
        def normalize_rating(r):
            return (r - self.min_rating) / self.rating_range
        
        # Convert to normalized ratings
        normalized_ratings_data = [(u, i, normalize_rating(r)) for u, i, r in ratings_data]
        
        # Train using SGD with momentum
        print("Training PMF model using Stochastic Gradient Descent...")
        
        current_lr = self.lr
        momentum = 0.9
        
        # User and item factor momentum terms
        user_momentum = np.zeros_like(self.user_factors)
        item_momentum = np.zeros_like(self.item_factors)
        
        # Pre-create a shuffled list of ratings for each epoch
        shuffled_data = list(normalized_ratings_data)
        
        # Create dictionary for faster lookup during training
        ratings_dict = {}
        for user_id, item_id, rating in normalized_ratings_data:
            if user_id not in ratings_dict:
                ratings_dict[user_id] = {}
            ratings_dict[user_id][item_id] = rating
        
        # Train over multiple epochs
        for epoch in range(self.n_epochs):
            # Shuffle the data each epoch
            random.shuffle(shuffled_data)
            
            squared_error = 0
            
            for user_id, item_id, norm_rating in shuffled_data:
                # Skip if user or item not in our mappings
                if user_id not in self.user_id_to_index or item_id not in self.item_id_to_index:
                    continue
                    
                # Get indices
                user_idx = self.user_id_to_index[user_id]
                item_idx = self.item_id_to_index[item_id]
                
                # Compute prediction (normalized)
                pred = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
                
                # Clip prediction to [0, 1] range
                pred = max(0, min(1, pred))
                
                # Calculate prediction error
                error = norm_rating - pred
                squared_error += error ** 2
                
                # Save old factors for update calculations
                old_user_factors = self.user_factors[user_idx].copy()
                old_item_factors = self.item_factors[item_idx].copy()
                
                # Calculate gradients with regularization
                user_gradient = -error * old_item_factors + self.user_reg * old_user_factors
                item_gradient = -error * old_user_factors + self.item_reg * old_item_factors
                
                # Update with momentum
                user_momentum[user_idx] = momentum * user_momentum[user_idx] - current_lr * user_gradient
                item_momentum[item_idx] = momentum * item_momentum[item_idx] - current_lr * item_gradient
                
                # Apply updates
                self.user_factors[user_idx] += user_momentum[user_idx]
                self.item_factors[item_idx] += item_momentum[item_idx]
            
            # Print epoch progress - converting RMSE back to original scale
            rmse = math.sqrt(squared_error / len(shuffled_data)) * self.rating_range
            print(f"Epoch {epoch+1}/{self.n_epochs}: RMSE = {rmse:.4f}, Learning Rate = {current_lr:.6f}")
            
            # Decay learning rate
            current_lr = max(self.min_lr, current_lr * self.lr_decay)
            
            # Early stopping (optional)
            if rmse < 0.01:  # Normalized RMSE threshold
                print(f"Converged early at epoch {epoch+1}!")
                break
        
        print("PMF training complete!")
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
        
        # Clip to valid rating range
        prediction = max(self.min_rating, min(self.max_rating, prediction))
        
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
    
    def calculate_posterior_variance(self):
        """
        Calculate the posterior variance of user and item factors
        This is specific to PMF's probabilistic interpretation
        """
        # For PMF, posterior variance is related to the regularization parameters
        user_variance = 1.0 / self.user_reg if self.user_reg > 0 else float('inf')
        item_variance = 1.0 / self.item_reg if self.item_reg > 0 else float('inf')
        
        return {
            'user_variance': user_variance,
            'item_variance': item_variance
        }


class AdaptedPMF(AlgoBase):
    """
    Adapter class that wraps PurePMF to make it compatible with Surprise's AlgoBase
    for evaluation with the Framework
    """
    
    def __init__(self, n_factors=10, n_epochs=50, lr=0.005, user_reg=0.1, item_reg=0.1, 
                 lr_decay=0.95, min_lr=0.001, init_std=0.1):
        """Initialize PMF parameters"""
        AlgoBase.__init__(self)
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.user_reg = user_reg
        self.item_reg = item_reg
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.init_std = init_std
        
        self.pure_pmf = PurePMF(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr=lr,
            user_reg=user_reg,
            item_reg=item_reg,
            lr_decay=lr_decay,
            min_lr=min_lr,
            init_std=init_std
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
        
        # Train our pure Python PMF implementation
        self.pure_pmf.fit(ratings_data)
        
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
            
            # Get prediction from pure PMF
            prediction = self.pure_pmf.predict(user_id, item_id)
            
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
    pmf_recommender = AdaptedPMF(n_factors=10, n_epochs=50, lr=0.005, 
                                 user_reg=0.1, item_reg=0.1, lr_decay=0.95)
    evaluator.AddAlgorithm(pmf_recommender, "PMF")
    
    random_predictor = NormalPredictor()
    evaluator.AddAlgorithm(random_predictor, "Random")
    
    # Run evaluation
    print("Evaluating algorithms...")
    evaluator.Evaluate(False)
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)