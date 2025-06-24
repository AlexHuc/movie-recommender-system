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
import numpy as np
import math
from collections import defaultdict

# Libs used for AdaptedALS
from surprise import AlgoBase, PredictionImpossible

# Libs to save and load models
from datetime import datetime
import pickle

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
        self.n_factors = n_factors                       # -> Number of latent factors (dimensionality of factor vectors)
        self.n_epochs = n_epochs                         # -> Number of alternating iterations
        self.reg = reg                                   # -> Regularization parameter (controls overfitting)
        self.confidence_scaling = confidence_scaling     # -> Scaling factor for confidence calculation
        
        # These will be initialized during fitting
        self.user_factors = None                         # -> Will store the user latent factor matrix (users × factors)
        self.item_factors = None                         # -> Will store the item latent factor matrix (items × factors)
        self.global_mean = 0                             # -> Global average of all ratings
        
        # Mappings for user and item IDs
        self.user_id_to_index = {}                       # -> Dictionary of user_id -> index
        self.item_id_to_index = {}                       # -> Dictionary of item_id -> index
        self.user_index_to_id = {}                       # -> Dictionary of index -> user_id
        self.item_index_to_id = {}                       # -> Dictionary of index -> item_id
        
        # Store ratings in structure optimized for ALS updates
        self.user_items = defaultdict(dict)              # -> Dictionary mapping user_idx -> {item_idx -> rating/confidence}
        self.item_users = defaultdict(dict)              # -> Dictionary mapping item_idx -> {user_idx -> rating/confidence}
    
    def fit(self, ratings_data):
        """
        Train the ALS model
        
        Args:
            ratings_data: List of (user_id, item_id, rating) tuples
        """

        """
        Example of the ALS model with a mini dataset:
            User 101: Movie 201 (5), Movie 202 (3), Movie 204 (4)
            User 102: Movie 201 (4), Movie 203 (5), Movie 205 (3)
            User 103: Movie 202 (4), Movie 203 (2), Movie 204 (3)

        ratings_data = [
            (101, 201, 5),  # User 101 rated Movie 201 with 5 stars
            (101, 202, 3),  # User 101 rated Movie 202 with 3 stars
            (101, 204, 4),  # User 101 rated Movie 204 with 4 stars
            (102, 201, 4),  # User 102 rated Movie 201 with 4 stars
            (102, 203, 5),  # User 102 rated Movie 203 with 5 stars
            (102, 205, 3),  # User 102 rated Movie 205 with 3 stars
            (103, 202, 4),  # User 103 rated Movie 202 with 4 stars
            (103, 203, 2),  # User 103 rated Movie 203 with 2 stars
            (103, 204, 3)   # User 103 rated Movie 204 with 3 stars
        ]

        # First, we process the data:
        self.users = [101, 102, 103]
        self.items = [201, 202, 203, 204, 205]
        self.user_id_to_index = {101: 0, 102: 1, 103: 2}
        self.item_id_to_index = {201: 0, 202: 1, 203: 2, 204: 3, 205: 4}
        self.user_index_to_id = {0: 101, 1: 102, 2: 103}
        self.item_index_to_id = {0: 201, 1: 202, 2: 203, 3: 204, 4: 205}
        
        # Calculate global mean rating
        self.global_mean = (5 + 3 + 4 + 4 + 5 + 3 + 4 + 2 + 3) / 9 = 3.67
        
        # Initialize factor matrices
        n_users = 3  # Number of unique users
        n_items = 5  # Number of unique items
        
        # Random initialization of factor matrices
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        # Example user factors (with 2 factors for simplicity):
        self.user_factors = np.array([
            [0.05, -0.08],  # User 101
            [0.09, 0.12],   # User 102
            [-0.03, 0.07]   # User 103
        ])

        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        # Example item factors:
        self.item_factors = np.array([
            [0.06, 0.10],   # Movie 201
            [-0.02, 0.08],  # Movie 202
            [0.09, -0.05],  # Movie 203
            [0.04, 0.09],   # Movie 204
            [-0.07, 0.06]   # Movie 205
        ])
        
        self.user_items = {
            0: {0: 5, 1: 3, 3: 4},      # User 101 ratings
            1: {0: 4, 2: 5, 4: 3},      # User 102 ratings
            2: {1: 4, 2: 2, 3: 3}       # User 103 ratings
        }
        self.item_users = {
            0: {0: 5, 1: 4},            # Movie 201 ratings
            1: {0: 3, 2: 4},            # Movie 202 ratings
            2: {1: 5, 2: 2},            # Movie 203 ratings
            3: {0: 4, 2: 3},            # Movie 204 ratings
            4: {1: 3}                   # Movie 205 ratings
        }

        min_rating = 2  # Minimum rating in the dataset
        max_rating = 5  # Maximum rating in the dataset
        self.rating_range = max_rating - min_rating = 5 - 2 = 3  # Range of ratings
        
        # Update the user_items and item_users with confidence values
        self.user_items = {
            0: { 0: 1.0 + 40 * (5 - 2) / 3, 1: 1.0 + 40 * (3 - 2) / 3, 3: 1.0 + 40 * (4 - 2) / 3}, # User 101
            1: { 0: 1.0 + 40 * (4 - 2) / 3, 2: 1.0 + 40 * (5 - 2) / 3, 4: 1.0 + 40 * (3 - 2) / 3}, # User 102
            2: { 1: 1.0 + 40 * (4 - 2) / 3, 2: 1.0 + 40 * (2 - 2) / 3, 3: 1.0 + 40 * (3 - 2) / 3}  # User 103
        } = {
            0: {0: 41, 1: 14, 3: 27}, # User 101 confidence values
            1: {0: 27, 2: 41, 4: 14}, # User 102 confidence values
            2: {1: 27, 2: 1, 3: 14}   # User 103 confidence values
        }

        # Update item_users with confidence values
        self.item_users = {
            0: {0: 1.0 + 40 * (5 - 2) / 3, 1: 1.0 + 40 * (4 - 2) / 3}, # Movie 201
            1: {0: 1.0 + 40 * (3 - 2) / 3, 2: 1.0 + 40 * (4 - 2) / 3}, # Movie 202
            2: {1: 1.0 + 40 * (5 - 2) / 3, 2: 1.0 + 40 * (2 - 2) / 3}, # Movie 203
            3: {0: 1.0 + 40 * (4 - 2) / 3, 2: 1.0 + 40 * (3 - 2) / 3}, # Movie 204
            4: {1: 1.0 + 40 * (3 - 2) / 3}                             # Movie 205
        } = {
            0: {0: 41, 1: 27},          # Movie 201 confidence values
            1: {0: 14, 2: 27},          # Movie 202 confidence values
            2: {1: 41, 2: 1 },          # Movie 203 confidence values
            3: {0: 27, 2: 14},          # Movie 204 confidence values
            4: {1: 14}                  # Movie 205 confidence values
        }

        # Initialize identity matrix scaled by regularization (reg * I)
        PP n_factors = 2  # Number of latent factors for simplicity
        reg = 0.1
        reg_I = [
          [0.1, 0.0],
          [0.0, 0.1]
        ]
        
        # ALS iterations (alternating between users and items)
        
        # Let's work through one iteration for User 0 (User 101)
        # Step 1: Fix item factors, solve for user factors
        
        # For User 0, the rated items are 0, 1, and 3 with confidence values 41.0, 14.2, 27.7
        
        # Build Y^T * C_u * Y + λI
        # Y^T * C_u * Y = sum over i (confidence_i * item_factors_i * item_factors_i^T)
        Y_CuY = reg_I.copy()  # Start with λI
        # Y_CuY = [
        #   [0.1, 0.0],
        #   [0.0, 0.1]
        # ]
        
        Y_Cu_ratings = np.zeros(self.n_factors)
        # Y_Cu_ratings = [0.0, 0.0]
        
        # For item 0 (Movie 201) with confidence 41.0
        item_factors = self.item_factors[0]  # [0.06, 0.10]
        Y_CuY += 41.0 * np.outer(item_factors, item_factors)
        # np.outer([0.06, 0.10], [0.06, 0.10]) = [
        #   [0.0036, 0.006],
        #   [0.006, 0.010]
        # ]
        # 41.0 * above = [
        #   [0.1476, 0.246],
        #   [0.246, 0.41]
        # ]
        # Y_CuY += above = [
        #   [0.1 + 0.1476, 0.0 + 0.246],
        #   [0.0 + 0.246, 0.1 + 0.41]
        # ] = [
        #   [0.2476, 0.246],
        #   [0.246, 0.51]
        # ]
        
        # Similarly for items 1 and 3, updating Y_CuY 
        
        # Build Y^T * C_u * p(u)
        # p(u) is the vector of ratings from user u
        # For item 0 (Movie 201) with confidence 41.0
        Y_Cu_ratings += 41.0 * item_factors
        # Y_Cu_ratings += 41.0 * [0.06, 0.10] = [41.0*0.06, 41.0*0.10] = [2.46, 4.10]
        
        # Similarly for items 1 and 3, updating Y_Cu_ratings
        
        # Solve the linear system: Y_CuY * x = Y_Cu_ratings
        # This gives us the new user factors
        # user_factors[0] = np.linalg.solve(Y_CuY, Y_Cu_ratings)
        
        # After solving for all users, we now fix user factors and solve for item factors
        # The process is symmetrical, now building X^T * C_i * X + λI
        # and X^T * C_i * p(i) for each item, then solving
        
        # After several iterations, the factor matrices converge
        # The key difference from SGD is that we're solving exact linear systems
        # at each iteration, rather than making small gradient steps.

        # Explanation of ALS:
        # The ratings matrix R would be:

        R = [
            [5, 3, ?, 4, ?],  # User 101
            [4, ?, 5, ?, 3],  # User 102
            [?, 4, 2, 3, ?]   # User 103
        ]

        ALS approximates this as the product of three matrices: R ≈ U * V^T, where:
        U = [
            [0.05, -0.08],  # User 101 factors
            [0.09, 0.12],   # User 102 factors
            [-0.03, 0.07]   # User 103 factors
        ]
        V = [
            [0.06, 0.10],   # Movie 201 factors
            [-0.02, 0.08],  # Movie 202 factors
            [0.09, -0.05],  # Movie 203 factors
            [0.04, 0.09],   # Movie 204 factors
            [-0.07, 0.06]   # Movie 205 factors
        ]
        
        """
        print("Processing ratings data for ALS matrix factorization...")
        
        # Extract all users, items, and ratings
        users = set()                                         # -> Create a set to track unique user IDs
        items = set()                                         # -> Create a set to track unique item IDs
        ratings_list = []                                     # -> Create a list to store all rating values
        
        for user_id, item_id, rating in ratings_data:         # -> Iterate through each (user_id, item_id, rating) tuple
            users.add(user_id)                                # -> Add user_id to the users set
            items.add(item_id)                                # -> Add item_id to the items set
            ratings_list.append(rating)                       # -> Append rating to the ratings list
        
        # Create mappings for users and items
        self.users = sorted(list(users))                      # -> Convert users set to a sorted list
        self.items = sorted(list(items))                      # -> Convert items set to a sorted list
        
        # Create dictionary mappings between IDs and indices
        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(self.users)}         # -> Create mapping from user_id to index
        self.item_id_to_index = {item_id: idx for idx, item_id in enumerate(self.items)}         # -> Create mapping from item_id to index
        
        self.user_index_to_id = {idx: user_id for user_id, idx in self.user_id_to_index.items()} # -> Create mapping from index to user_id
        self.item_index_to_id = {idx: item_id for item_id, idx in self.item_id_to_index.items()} # -> Create mapping from index to item_id
        
        # Calculate global mean rating
        self.global_mean = sum(ratings_list) / len(ratings_list) if ratings_list else 0          # -> Calculate average of all ratings
        
        # Initialize parameters
        n_users = len(self.users)                             # -> Number of unique users
        n_items = len(self.items)                             # -> Number of unique items
        
        # Initialize factor matrices with small random values
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))                  # -> Initialize user factor matrix with random values
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))                  # -> Initialize item factor matrix with random values
        
        # Build user-item and item-user mappings for efficient lookups during ALS
        for user_id, item_id, rating in ratings_data:         # -> Iterate through all ratings to build the mappings
            if user_id in self.user_id_to_index and item_id in self.item_id_to_index:
                user_idx = self.user_id_to_index[user_id]     # -> Convert user_id to matrix index
                item_idx = self.item_id_to_index[item_id]     # -> Convert item_id to matrix index
                
                # Store ratings in both directions for efficient lookups
                self.user_items[user_idx][item_idx] = rating  # -> Map user to their ratings
                self.item_users[item_idx][user_idx] = rating  # -> Map item to its ratings
        
        # Normalize ratings to [0, 1] for confidence calculation
        min_rating = min(ratings_list)                        # -> Find minimum rating value
        max_rating = max(ratings_list)                        # -> Find maximum rating value
        self.rating_range = max_rating - min_rating           # -> Calculate range of ratings
        self.min_rating = min_rating                          # -> Store for later denormalization
        
        # Convert ratings to confidence values
        # Key concept in ALS: Higher ratings = more confidence in preference
        for user_idx, items in self.user_items.items():       # -> Process all user ratings
            for item_idx, rating in items.items():            # -> Process each rated item
                # Normalize rating to [0, 1]
                norm_rating = (rating - min_rating) / self.rating_range if self.rating_range > 0 else 0.5  # -> Scale to [0,1]
                
                # Calculate confidence: 1 + scaling_factor * normalized_rating
                # A rating of min_rating gives confidence of 1.0
                # A rating of max_rating gives confidence of 1.0 + confidence_scaling
                confidence = 1.0 + self.confidence_scaling * norm_rating  # -> Convert to confidence value
                
                # Update with confidence value instead of rating
                self.user_items[user_idx][item_idx] = confidence  # -> Replace rating with confidence in user map
                self.item_users[item_idx][user_idx] = confidence  # -> Replace rating with confidence in item map
        
        # Start ALS iterations
        print("Training ALS model using alternating least squares...")
        
        # Identity matrix scaled by regularization parameter (I * λ)
        # This will be added to all systems to prevent overfitting
        reg_I = self.reg * np.eye(self.n_factors)       # -> Create regularization matrix
        
        for epoch in range(self.n_epochs):              # -> Iterate through each training epoch
            print(f"Epoch {epoch+1}/{self.n_epochs}")
            
            # Step 1: Fix item factors, solve for user factors
            print("  Updating user factors...")
            
            for u_idx in range(n_users):                # -> Iterate through each user
                if u_idx % 100 == 0:                    # -> Print progress every 100 users
                    print(f"  Processing user {u_idx}/{n_users}")
                    
                # Get all items rated by this user
                rated_items = self.user_items[u_idx]    # -> Get this user's ratings
                
                if not rated_items:                     # -> Skip if user has no ratings
                    continue
                
                # Build Y^T * C_u * Y + λI
                # Y is the item factors matrix
                # C_u is a diagonal matrix of confidence values for user u
                Y_CuY = reg_I.copy()                    # -> Start with regularization matrix
                Y_Cu_ratings = np.zeros(self.n_factors) # -> Initialize ratings vector
                
                for i_idx, confidence in rated_items.items():  # -> Process each rated item
                    # Get item factors (Y_i)
                    item_factors = self.item_factors[i_idx]    # -> Get factor vector for this item
                    
                    # Update Y^T * C_u * Y
                    # This is the weighted outer product of item vectors
                    # weighted by the confidence
                    Y_CuY += confidence * np.outer(item_factors, item_factors)  # -> Accumulate weighted outer products
                    
                    # Update Y^T * C_u * p(u)
                    # This is the item vector weighted by confidence and observed rating
                    # For implicit feedback, the "rating" is 1.0 for observed interactions
                    # For explicit feedback, we can treat rating as 1.0 and incorporate the
                    # actual rating value into the confidence
                    Y_Cu_ratings += confidence * item_factors  # -> Accumulate weighted item factors
                
                # Solve the linear system Y_CuY * x = Y_Cu_ratings
                # This gives us the optimal user factor vector
                try:
                    # Solve linear system for this user's factors
                    self.user_factors[u_idx] = np.linalg.solve(Y_CuY, Y_Cu_ratings)  # -> Solve for optimal user factors
                except np.linalg.LinAlgError:
                    # Fallback for singular matrices
                    # Use least squares to find approximate solution
                    self.user_factors[u_idx], _, _, _ = np.linalg.lstsq(Y_CuY, Y_Cu_ratings, rcond=None)  # -> Fallback to least squares
            
            # Step 2: Fix user factors, solve for item factors
            print("  Updating item factors...")
            
            for i_idx in range(n_items):                # -> Iterate through each item
                if i_idx % 100 == 0:                    # -> Print progress every 100 items
                    print(f"  Processing item {i_idx}/{n_items}")
                    
                # Get all users who rated this item
                rated_users = self.item_users[i_idx]    # -> Get this item's ratings
                
                if not rated_users:                     # -> Skip if item has no ratings
                    continue
                
                # Build X^T * C_i * X + λI
                # X is the user factors matrix
                # C_i is a diagonal matrix of confidence values for item i
                X_CiX = reg_I.copy()                    # -> Start with regularization matrix
                X_Ci_ratings = np.zeros(self.n_factors) # -> Initialize ratings vector
                
                for u_idx, confidence in rated_users.items():  # -> Process each user who rated this item
                    # Get user factors (X_u)
                    user_factors = self.user_factors[u_idx]    # -> Get factor vector for this user
                    
                    # Update X^T * C_i * X
                    X_CiX += confidence * np.outer(user_factors, user_factors)  # -> Accumulate weighted outer products
                    
                    # Update X^T * C_i * p(i)
                    X_Ci_ratings += confidence * user_factors  # -> Accumulate weighted user factors
                
                # Solve the linear system X_CiX * x = X_Ci_ratings
                # This gives us the optimal item factor vector
                try:
                    # Solve linear system for this item's factors
                    self.item_factors[i_idx] = np.linalg.solve(X_CiX, X_Ci_ratings)  # -> Solve for optimal item factors
                except np.linalg.LinAlgError:
                    # Fallback for singular matrices
                    self.item_factors[i_idx], _, _, _ = np.linalg.lstsq(X_CiX, X_Ci_ratings, rcond=None)  # -> Fallback to least squares
            
            # Calculate and print RMSE occasionally
            if epoch % 2 == 0:  # -> Calculate metrics every other epoch
                rmse = self._compute_rmse(ratings_data)  # -> Calculate RMSE on training data
                print(f"  RMSE on training data: {rmse:.4f}")
        
        print("ALS training complete!")
        return self
    
    def _compute_rmse(self, ratings_data, sample_size=10000):
        """
        Compute RMSE on a sample of ratings data
        
        Args:
            ratings_data: List of (user_id, item_id, rating) tuples
            sample_size: Maximum number of samples to use
            
        Returns:
            RMSE value
        """
        # Sample data if too large
        if len(ratings_data) > sample_size:  # -> Use sampling for large datasets
            sampled_data = random.sample(ratings_data, sample_size)  # -> Take a random sample
        else:
            sampled_data = ratings_data  # -> Use all data if small enough
            
        squared_error_sum = 0.0  # -> Track sum of squared errors
        count = 0  # -> Track number of valid predictionss
        
        for user_id, item_id, true_rating in sampled_data:  # -> Iterate through sampled data
            if user_id not in self.user_id_to_index or item_id not in self.item_id_to_index:
                continue  # -> Skip if user or item not in training data
                
            pred_rating = self.predict(user_id, item_id)  # -> Get prediction
            squared_error_sum += (true_rating - pred_rating) ** 2  # -> Calculate squared error
            count += 1  # -> Increment count
            
        if count == 0:  # -> Handle edge case of no valid predictions
            return float('inf')
            
        return math.sqrt(squared_error_sum / count)  # -> Calculate RMSE
    
    def predict(self, user_id, item_id):
        """
        Predict rating for a user-item pair
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        """
        Example:
            For User 101 and Movie 203:
            
            user_idx = 0 (index for User 101)
            item_idx = 2 (index for Movie 203)
            
            user_factors[0] = [0.05, -0.08]  
            item_factors[2] = [0.09, -0.05]
            
            # Calculate dot product
            dot_product = np.dot([0.05, -0.08], [0.09, -0.05])
            dot_product = 0.05 * 0.09 + (-0.08) * (-0.05)
            dot_product = 0.0045 + 0.004
            dot_product = 0.0085
            
            # If this represents confidence, convert back to rating
            # Assuming confidence_scaling = 40:
            # If predicted confidence is 0.0085:
            # 1. First subtract 1.0 to get relative confidence: 0.0085 - 1.0 = -0.9915
            # 2. Since negative, clamp to 0: max(0, -0.9915) = 0
            # 3. Divide by confidence_scaling: 0 / 40 = 0
            # 4. Scale back to rating range and add min_rating:
            #    0 * 3 + 2 = 2
            
            # So the prediction would be 2 (minimum rating)
            # Note: In practice, dot product in ALS often is higher and results in ratings
            # across the whole rating scale, this is just a simplified example
        """
        
        # Check if user and item exist in the training data
        if user_id not in self.user_id_to_index or item_id not in self.item_id_to_index:
            # For cold start, return global mean
            return self.global_mean  # -> Return global mean for unknown users/items
        
        # Get indices
        user_idx = self.user_id_to_index[user_id]  # -> Convert user_id to matrix index
        item_idx = self.item_id_to_index[item_id]  # -> Convert item_id to matrix index
        
        # Compute prediction as dot product of user and item factors
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])  # -> Calculate dot product
        
        # Scale back from confidence to rating
        # Subtract 1.0 because confidence = 1.0 + scaling * normalized_rating
        scaled_prediction = (prediction - 1.0) / self.confidence_scaling if prediction > 1.0 else 0.0  # -> Convert from confidence to normalized rating
        
        # Convert back to original rating scale
        original_prediction = scaled_prediction * self.rating_range + self.min_rating  # -> Convert from normalized to original rating scale
        
        # Clip to valid rating range (typically 1-5)
        original_prediction = max(self.min_rating, min(self.min_rating + self.rating_range, original_prediction))  # -> Ensure rating is within valid range
        
        return original_prediction  # -> Return the predicted rating
    
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

    def save_model(self, filename=None):
        """
        Save the trained model to disk
        
        Args:
            filename: Path to save the model (if None, a default name will be generated)
            
        Returns:
            str: Path where model was saved
        """
        if filename is None:
            # Create a default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"../models/2_MatrixFactorization/adapted_als_model_{timestamp}.pkl"
        
        # Ensure the models directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Save the model using pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Adapted model saved to {filename}")
        return filename

    @classmethod
    def load_model(cls, filename):
        """
        Load a trained model from disk
        
        Args:
            filename: Path to the saved model file
            
        Returns:
            AdaptedUserKNN: Loaded model instance
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file {filename} not found")
        
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Adapted model loaded from {filename}")
        return model

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

    """
    | Agorithms                       | RMSE   | MAE       |
    |---------------------------------|--------|-----------|
    | Random                          | 1.4385 |    1.1478 |
    | Collaborative Filtering         |--------|-----------|
    | ALS                             | 0.9053 |    0.6987 |   
    """
    
    # Run evaluation
    print("Evaluating algorithms...")
    evaluator.Evaluate(False)

    als_recommender.save_model()
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)