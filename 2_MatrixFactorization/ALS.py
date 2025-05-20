# -*- coding: utf-8 -*-
"""
Custom Alternating Least Squares (ALS) Algorithm Implementation 
without Surprise but compatible with Framework evaluation
"""
import sys
import os
# Get the absolute path to the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Libs used for PureALS
from EvaluationFramework.MovieLens import MovieLens
import numpy as np
import math
from collections import defaultdict

# Libs used for AdaptedALS
from surprise import AlgoBase, PredictionImpossible

# Libs used for Evaluation
from utils.LoadMovieLensData import LoadMovieLensData
from EvaluationFramework.Evaluator import Evaluator
from surprise import NormalPredictor
import random


class PureALS:
    """
    A pure Python implementation of Alternating Least Squares (ALS) for matrix factorization
    
    ALS alternates between fixing user factors and solving for item factors, and vice versa,
    which allows for efficient parallelization and handles implicit feedback well.
    """
    
    def __init__(self, n_factors=20, n_epochs=15, reg=0.1, confidence_scaling=40):
        """
        Initialize ALS parameters
        
        Args:
            n_factors (int): Number of latent factors
            n_epochs (int): Number of iterations
            reg (float): Regularization term
            confidence_scaling (float): Scaling factor for implicit confidence
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg  # Regularization factor
        self.confidence_scaling = confidence_scaling
        
        # These will be initialized during fitting
        self.user_factors = None
        self.item_factors = None
        self.global_mean = 0
        
        # Mappings for user and item IDs
        self.user_id_to_index = {}
        self.item_id_to_index = {}
        self.user_index_to_id = {}
        self.item_index_to_id = {}
        
        # Store ratings in structure optimized for ALS updates
        self.user_items = defaultdict(dict)  # user_idx -> {item_idx -> rating}
        self.item_users = defaultdict(dict)  # item_idx -> {user_idx -> rating}
    
    def fit(self, ratings_data):
        """
        Train the ALS model
        
        Args:
            ratings_data: List of (user_id, item_id, rating) tuples
        """
        print("Processing ratings data for ALS matrix factorization...")
        
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
        
        # Initialize factor matrices
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # Build user-item and item-user mappings
        for user_id, item_id, rating in ratings_data:
            if user_id in self.user_id_to_index and item_id in self.item_id_to_index:
                user_idx = self.user_id_to_index[user_id]
                item_idx = self.item_id_to_index[item_id]
                
                # Store ratings in both directions for efficient lookups
                self.user_items[user_idx][item_idx] = rating
                self.item_users[item_idx][user_idx] = rating
        
        # Normalize ratings to [0, 1] for confidence calculation
        min_rating = min(ratings_list)
        max_rating = max(ratings_list)
        self.rating_range = max_rating - min_rating
        self.min_rating = min_rating
        
        # Create confidence matrix from ratings
        # Higher ratings = more confidence
        for user_idx, items in self.user_items.items():
            for item_idx, rating in items.items():
                # Normalize rating to [0, 1]
                norm_rating = (rating - min_rating) / self.rating_range if self.rating_range > 0 else 0.5
                
                # Calculate confidence: 1 + scaling_factor * normalized_rating
                confidence = 1.0 + self.confidence_scaling * norm_rating
                
                # Update with confidence value instead of rating
                self.user_items[user_idx][item_idx] = confidence
                self.item_users[item_idx][user_idx] = confidence
        
        # Start ALS iterations
        print("Training ALS model using alternating least squares...")
        
        # Identity matrix scaled by regularization parameter (I * λ)
        reg_I = self.reg * np.eye(self.n_factors)
        
        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch+1}/{self.n_epochs}")
            
            # Step 1: Fix item factors, solve for user factors
            print("  Updating user factors...")
            
            for u_idx in range(n_users):
                if u_idx % 100 == 0:
                    print(f"  Processing user {u_idx}/{n_users}")
                    
                # Get all items rated by this user
                rated_items = self.user_items[u_idx]
                
                if not rated_items:
                    continue
                
                # Build Y^T * C_u * Y + λI
                Y_CuY = reg_I.copy()
                Y_Cu_ratings = np.zeros(self.n_factors)
                
                for i_idx, confidence in rated_items.items():
                    # Get item factors (Y_i)
                    item_factors = self.item_factors[i_idx]
                    
                    # Update Y^T * C_u * Y
                    Y_CuY += confidence * np.outer(item_factors, item_factors)
                    
                    # Update Y^T * C_u * p(u)
                    Y_Cu_ratings += confidence * item_factors
                
                # Solve the linear system
                try:
                    self.user_factors[u_idx] = np.linalg.solve(Y_CuY, Y_Cu_ratings)
                except np.linalg.LinAlgError:
                    # Fallback for singular matrices
                    self.user_factors[u_idx], _, _, _ = np.linalg.lstsq(Y_CuY, Y_Cu_ratings, rcond=None)
            
            # Step 2: Fix user factors, solve for item factors
            print("  Updating item factors...")
            
            for i_idx in range(n_items):
                if i_idx % 100 == 0:
                    print(f"  Processing item {i_idx}/{n_items}")
                    
                # Get all users who rated this item
                rated_users = self.item_users[i_idx]
                
                if not rated_users:
                    continue
                
                # Build X^T * C_i * X + λI
                X_CiX = reg_I.copy()
                X_Ci_ratings = np.zeros(self.n_factors)
                
                for u_idx, confidence in rated_users.items():
                    # Get user factors (X_u)
                    user_factors = self.user_factors[u_idx]
                    
                    # Update X^T * C_i * X
                    X_CiX += confidence * np.outer(user_factors, user_factors)
                    
                    # Update X^T * C_i * p(i)
                    X_Ci_ratings += confidence * user_factors
                
                # Solve the linear system
                try:
                    self.item_factors[i_idx] = np.linalg.solve(X_CiX, X_Ci_ratings)
                except np.linalg.LinAlgError:
                    # Fallback for singular matrices
                    self.item_factors[i_idx], _, _, _ = np.linalg.lstsq(X_CiX, X_Ci_ratings, rcond=None)
            
            # Calculate and print RMSE
            if epoch % 2 == 0:
                rmse = self._compute_rmse(ratings_data)
                print(f"  RMSE on training data: {rmse:.4f}")
        
        print("ALS training complete!")
        return self
    
    def _compute_rmse(self, ratings_data, sample_size=10000):
        """Compute RMSE on a sample of ratings data"""
        if len(ratings_data) > sample_size:
            sampled_data = random.sample(ratings_data, sample_size)
        else:
            sampled_data = ratings_data
            
        squared_error_sum = 0.0
        count = 0
        
        for user_id, item_id, true_rating in sampled_data:
            if user_id not in self.user_id_to_index or item_id not in self.item_id_to_index:
                continue
                
            pred_rating = self.predict(user_id, item_id)
            squared_error_sum += (true_rating - pred_rating) ** 2
            count += 1
            
        if count == 0:
            return float('inf')
            
        return math.sqrt(squared_error_sum / count)
    
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
        
        # Compute prediction as dot product of user and item factors
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        # Scale back from confidence to rating
        scaled_prediction = (prediction - 1.0) / self.confidence_scaling if prediction > 1.0 else 0.0
        
        # Convert back to original rating scale
        original_prediction = scaled_prediction * self.rating_range + self.min_rating
        
        # Clip to valid rating range (typically 1-5)
        original_prediction = max(self.min_rating, min(self.min_rating + self.rating_range, original_prediction))
        
        return original_prediction
    
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


class AdaptedALS(AlgoBase):
    """
    Adapter class that wraps PureALS to make it compatible with Surprise's AlgoBase
    for evaluation with the Framework
    """
    
    def __init__(self, n_factors=20, n_epochs=15, reg=0.1, confidence_scaling=40):
        """Initialize ALS parameters"""
        AlgoBase.__init__(self)
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg
        self.confidence_scaling = confidence_scaling
        self.pure_als = PureALS(
            n_factors=n_factors, 
            n_epochs=n_epochs, 
            reg=reg,
            confidence_scaling=confidence_scaling
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
        
        # Train our pure Python ALS implementation
        self.pure_als.fit(ratings_data)
        
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
            
            # Get prediction from pure ALS
            prediction = self.pure_als.predict(user_id, item_id)
            
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
    als_recommender = AdaptedALS(n_factors=20, n_epochs=15, reg=0.1, confidence_scaling=40)
    evaluator.AddAlgorithm(als_recommender, "ALS")
    
    random_predictor = NormalPredictor()
    evaluator.AddAlgorithm(random_predictor, "Random")
    
    # Run evaluation
    print("Evaluating algorithms...")
    evaluator.Evaluate(False)
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)