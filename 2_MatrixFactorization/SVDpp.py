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
        self.n_factors = n_factors                       # -> Number of latent factors to use in factorization (dimensionality of user/item vectors)
        self.n_epochs = n_epochs                         # -> Number of training iterations over the entire dataset
        self.lr = lr                                     # -> Learning rate for gradient descent (controls step size)
        self.reg = reg                                   # -> Regularization parameter (prevents overfitting)
        # New in SVD++
        self.implicit_weight = implicit_weight           # -> Weight to control the influence of implicit feedback
        
        # These will be initialized during fitting
        self.user_factors = None                         # -> Will store the user latent factor matrix (users × factors)
        self.item_factors = None                         # -> Will store the item latent factor matrix (items × factors)
        # New in SVD++
        self.item_implicit_factors = None                # -> Will store the item implicit factor matrix (items × factors)
        self.global_mean = 0                             # -> Global average of all ratings
        self.user_biases = {}                            # -> User bias terms (user_id -> bias)
        self.item_biases = {}                            # -> Item bias terms (item_id -> bias)
        
        # Mappings for user and item IDs
        self.user_id_to_index = {}                       # -> Dictionary of user_id -> index
        self.item_id_to_index = {}                       # -> Dictionary of item_id -> index
        self.user_index_to_id = {}                       # -> Dictionary of index -> user_id
        self.item_index_to_id = {}                       # -> Dictionary of index -> item_id
        
        # # New in SVD++: User-item interactions (implicit feedback)
        self.user_rated_items = defaultdict(list)        # -> Dictionary of user_id -> list of item indices the user has rated
        self.sqrt_user_rated_counts = {}                 # -> Dictionary of user_id -> 1/sqrt(number of items rated by user)
    
    def fit(self, ratings_data):
        """
        Train the SVD++ model using SGD
        
        Args:
            ratings_data: List of (user_id, item_id, rating) tuples
        """

        """
        Example of the SVD++ model with a mini dataset:
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

        self.users = [101, 102, 103]
        self.items = [201, 202, 203, 204, 205]
        self.user_id_to_index = {101: 0, 102: 1, 103: 2
        self.item_id_to_index = {201: 0, 202: 1, 203: 2, 204: 3, 205: 4}
        self.user_index_to_id = {0: 101, 1: 102, 2: 103}
        self.item_index_to_id = {0: 201, 1: 202, 2: 203, 3: 204, 4: 205}
        self.global_mean = (5 + 3 + 4 + 4 + 5 + 3 + 4 + 2 + 3) / 9  = 3.67

        n_users = 3  # Number of unique users
        n_items = 5  # Number of unique items

        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))   # Random initialization of user factors
        # Example user factors:
        self.user_factors = np.array([
            [0.5, 0.8],  # User 101
            [0.9, 0.2],  # User 102
            [0.3, 0.7]   # User 103
        ])

        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))   # Random initialization of item factors
        # Example item factors:
        self.item_factors = np.array([
            [0.6, 0.3],  # Movie 201
            [0.2, 0.8],  # Movie 202
            [0.8, 0.1],  # Movie 203
            [0.4, 0.9],  # Movie 204
            [0.1, 0.5]   # Movie 205
        ])

        self.item_implicit_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))  # Random initialization of item implicit factors
        # Example item implicit factors:
        self.item_implicit_factors = np.array([
            [0.4, 0.6],  # Movie 201
            [0.5, 0.2],  # Movie 202
            [0.3, 0.7],  # Movie 203
            [0.6, 0.1],  # Movie 204
            [0.2, 0.4]   # Movie 205
        ])

        self.user_biases = {101: 0.33, 102: 0.0, 103: -0.33}                      # Example user biases
        self.item_biases = {201: 0.33, 202: 0.0, 203: 0.0, 204: 0.0, 205: -0.33}  # Example item biases

        self.user_counts = {101: 3, 102: 3, 103: 3}                               # Count of ratings per user
        self.item_counts = {201: 2, 202: 2, 203: 2, 204: 2, 205: 1}               # Count of ratings per item
        
        # Key difference in SVD++ compared to SVD:
        # We track which items each user has rated (implicit feedback)

        self.user_rated_items = {
            101: [0, 1, 3],  # User 101 rated items at indices 0 (201), 1 (202), and 3 (204)
            102: [0, 2, 4],  # User 102 rated items at indices 0 (201), 2 (203), and 4 (205)
            103: [1, 2, 3]   # User 103 rated items at indices 1 (202), 2 (203), and 3 (204)
        }
        
        # Normalization factors for implicit feedback
        self.sqrt_user_rated_counts = {
            101: 1/sqrt(3) ≈ 0.577,  # User 101 rated 3 items
            102: 1/sqrt(3) ≈ 0.577,  # User 102 rated 3 items
            103: 1/sqrt(3) ≈ 0.577   # User 103 rated 3 items
        }
        
        # In SVD++, for each user, we compute an implicit feedback vector which is the sum
        # of the implicit item factors for all items the user has rated, divided by sqrt(number of items rated)

        shuffled_data = list(ratings_data)
        # How the shuffled_data looks like unsuffled:
        shuffled_data = [
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

        # First iteration of the training epoch:
        # How the shuffled_data looks like suffled:
        shuffled_data = [
            (102, 205, 3),  # User 102 rated Movie 205 with 3 stars
            (103, 203, 2),  # User 103 rated Movie 203 with 2 stars
            (101, 201, 5),  # User 101 rated Movie 201 with 5 stars
            (101, 202, 3),  # User 101 rated Movie 202 with 3 stars
            (102, 201, 4),  # User 102 rated Movie 201 with 4 stars
            (103, 202, 4),  # User 103 rated Movie 202 with 4 stars
            (101, 204, 4),  # User 101 rated Movie 204 with 4 stars
            (102, 203, 5),  # User 102 rated Movie 203 with 5 stars
            (103, 204, 3)   # User 103 rated Movie 204 with 3 stars
        ]

            #First iteration of the training epoch:
                sqrt_count = 0.577  # For User 102

                if sqrt_count > 0:
                    implicit_sum = ( self.item_implicit_factors[0] + self.item_implicit_factors[2] + self.item_implicit_factors[4]) * sqrt_count  =
                                 = ([0.4, 0.6] + [0.3, 0.7] + [0.2, 0.4]) * 0.577
                                 = [0.9, 1.7] * 0.577
                                 = [0.519, 0.981]  # Implicit feedback vector for User 102
                else:
                    implicit_sum = [0.0, 0.0]

                pred = global_mean + user_biases[102] + item_biases[205] + dot(user_factors[1] + implicit_weight * implicit_sum, item_factors[4]) =
                = 3.67 + 0.0 + -0.33 + ([0.9, 0.2] + 0.1 * [0.519, 0.981]) * [0.1, 0.5]
                = 3.67 + 0.0 - 0.33 + ([0.9, 0.2] + [0.0519, 0.0981]) * [0.1, 0.5]
                = 3.67 - 0.33 + [0.9519, 0.2981] * [0.1, 0.5]
                = 3.67 - 0.33 + (0.09519 + 0.14905)
                = 3.67 - 0.33 + 0.24424
                = 3.58176  # Predicted rating for User 102 on Movie 205

                error = rating - pred = 5 - 3.58176 = 1.41824  # Error for this prediction
                squared_error += error ** 2 = 1.41824 ** 2 = 2.0103  # Accumulate squared error for RMSE calculation

                user_biases[user_id] += lr * (error - reg * user_biases[user_id]) = 
                                     = 0.0 + 0.005 * (1.41824 - 0.02 * 0.0) 
                                     = 0.0 + 0.005 * 1.41824 
                                     = 0.0070912
                
                item_biases[item_id] += lr * (error - reg * item_biases[item_id]) =
                                     = -0.33 + 0.005 * (1.41824 - 0.02 * -0.33)
                                     = -0.33 + 0.005 * (1.41824 + 0.0066)
                                     = -0.33 + 0.005 * 1.42484
                                     = -0.33 + 0.0071242
                                     = -0.3228758
                
                user_factors[user_idx] += lr * (error * old_item_factors - reg * old_user_factors) =
                                        = [0.9, 0.2] + 0.005 * (1.41824 * [0.1, 0.5] - 0.02 * [0.9, 0.2])
                                        = [0.9, 0.2] + 0.005 * ([0.141824, 0.70912] - [0.018, 0.004])
                                        = [0.9, 0.2] + 0.005 * [0.123824, 0.70512]
                                        = [0.9, 0.2] + [0.00061912, 0.0035256]
                                        = [0.90061912, 0.2035256]
                
                item_factors[item_idx] += lr * (error * (old_user_factors + implicit_weight * implicit_sum) - reg * old_item_factors) =
                                        = [0.1, 0.5] + 0.005 * (1.41824 * ([0.9, 0.2] + 0.1 * [0.519, 0.981]) - 0.02 * [0.1, 0.5])
                                        = [0.1, 0.5] + 0.005 * (1.41824 * ([0.9 + 0.0519, 0.2 + 0.0981]) - [0.002, 0.01])
                                        = [0.1, 0.5] + 0.005 * (1.41824 * [0.9519, 0.2981] - [0.002, 0.01])
                                        = [0.1, 0.5] + 0.005 * ([0.135, 0.0443] - [0.002, 0.01])
                                        = [0.1, 0.5] + 0.005 * [0.133, 0.0343]
                                        = [0.1, 0.5] + [0.000665, 0.0001715]
                                        = [0.100665, 0.5001715] 
        
        On the current example the user_factors and item_factors matrices would look like this in the final:
        # User factors matrix (users × factors)
        self.user_factors = np.array([    
            [0.90123, 0.20526],  # User 101
            [0.90123, 0.20526],  # User 102
            [0.30123, 0.70526]   # User 103
        ])
        # Item factors matrix (items × factors)
        self.item_factors = np.array([
            [0.10123, 0.50526],  # Movie 201
            [0.20123, 0.80526],  # Movie 202
            [0.30123, 0.10526],  # Movie 203
            [0.40123, 0.90526],  # Movie 204
            [0.50123, 0.20526]   # Movie 205
        ])

        # Explanation of SVD:
        # The ratings matrix R would be:
        
        R = [
            [5, 3, ?, 4, ?],  # User 101
            [4, ?, 5, ?, 3],  # User 102
            [?, 4, 2, 3, ?]   # User 103
        ]

        SVD approximates this as the product of three matrices: R ≈ U * Σ * V^T
        Where:
            - U is the user factors matrix (users × factors)
            - Σ is the diagonal matrix of singular values (factors × factors)
            - V^T is the transposed item factors matrix (factors × items)
        The goal is to learn U and V such that the dot product U * V^T approximates R.

        For our implementation, we are using a variant called Funk SVDpp where:

        R ≈ global_mean + user_biases + item_biases + (U + implicit_weight * implicit_sum)·V^T
        
        Where: 
            - global_mean: The average rating across all users and items (e.g., 3.67)
            - user_biases: User-specific rating tendencies (e.g., {101: 0.33, 102: 0.0, 103: -0.33})
            - item_biases: Item-specific rating tendencies (e.g., {201: 0.33, 202: 0.0, 203: 0.0, 204: 0.0, 205: -0.33})
            - U: User factors matrix with n_factors columns (e.g., 3x2 for 3 users and 2 factors)
                [0.90123, 0.20526]
                [0.90123, 0.20526]
                [0.30123, 0.70526]
            - implicit_weight: A weight to control the influence of implicit feedback (e.g., 0.1)
            - implicit_sum: The sum of implicit item factors for items the user has rated, normalized by sqrt(number of items rated)
            - V: Item factors matrix with n_factors columns (e.g., 5x2 for 5 items and 2 factors)
                [0.10123, 0.50526]
                [0.20123, 0.80526]
                [0.30123, 0.10526]
                [0.40123, 0.90526]
                [0.50123, 0.20526]

        To predict a rating for User 101 on Movie 203:
        prediction = 3.67 + 0.33 + 0.0 + dot([0.5289, 0.8404], [0.8, 0.1])
                   = 4.0 + (0.5289*0.8 + 0.8404*0.1)
                   = 4.0 + 0.42312 + 0.08404
                   = 4.5072

        """
        print("Processing ratings data for SVD++ matrix factorization...")
        
        # Extract all users, items, and ratings
        users = set()                                     # -> Create a set to track unique user IDs
        items = set()                                     # -> Create a set to track unique item IDs
        ratings_list = []                                 # -> Create a list to store all rating values
        
        for user_id, item_id, rating in ratings_data:     # -> Iterate through each (user_id, item_id, rating) tuple
            users.add(user_id)                            # -> Add user_id to the users set
            items.add(item_id)                            # -> Add item_id to the items set
            ratings_list.append(rating)                   # -> Append rating to the ratings list
        
        # Create mappings for users and items
        self.users = sorted(list(users))                  # -> Convert users set to a sorted list
        self.items = sorted(list(items))                  # -> Convert items set to a sorted list
        
        # Create dictionary mappings between IDs and indices
        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(self.users)}    # -> Create mapping from user_id to index
        self.item_id_to_index = {item_id: idx for idx, item_id in enumerate(self.items)}    # -> Create mapping from item_id to index
        
        self.user_index_to_id = {idx: user_id for user_id, idx in self.user_id_to_index.items()} # -> Create mapping from index to user_id
        self.item_index_to_id = {idx: item_id for item_id, idx in self.item_id_to_index.items()} # -> Create mapping from index to item_id
        
        # Calculate global mean rating
        self.global_mean = sum(ratings_list) / len(ratings_list) if ratings_list else 0   # -> Calculate average of all ratings
        
        # Initialize parameters
        n_users = len(self.users)                        # -> Number of unique users
        n_items = len(self.items)                        # -> Number of unique items
        
        # User factors matrix (users × factors)
        # Each row represents a user's explicit latent factors, initialized with small random values
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))   # -> Initialize user factor matrix with random values
        
        # Item factors matrix (items × factors)
        # Each row represents an item's latent factors, initialized with small random values
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))   # -> Initialize item factor matrix with random values
        
        # Item implicit factors matrix (items × factors) - THIS IS UNIQUE TO SVD++
        # Each row represents an item's implicit latent factors, initialized with small random values
        self.item_implicit_factors = np.random.normal(0, 0.1, (n_items, self.n_factors)) # -> Initialize item implicit factor matrix with random values
        
        # Initialize biases to zero
        self.user_biases = {u: 0.0 for u in self.users}  # -> Initialize user biases to 0.0
        self.item_biases = {i: 0.0 for i in self.items}  # -> Initialize item biases to 0.0
        
        # Build user-item interaction data for implicit feedback
        print("Building implicit feedback data...")
        for user_id, item_id, _ in ratings_data:          # -> Iterate through all ratings to build implicit feedback data
            if user_id in self.user_id_to_index and item_id in self.item_id_to_index:
                # Add the item index to the list of items rated by this user
                item_idx = self.item_id_to_index[item_id]
                self.user_rated_items[user_id].append(item_idx) # -> Track which items each user has rated
        
        # Pre-calculate the sqrt of user rated items lengths (for normalization)
        # In SVD++, the implicit feedback component is normalized by the square root of the number of items rated
        self.sqrt_user_rated_counts = {}
        for user_id, rated_items in self.user_rated_items.items():
            # Calculate 1/sqrt(number of items rated) for each user
            self.sqrt_user_rated_counts[user_id] = 1.0 / math.sqrt(len(rated_items)) if rated_items else 0
        
        # Train using SGD (stochastic gradient descent)
        print("Training SVD++ model using Stochastic Gradient Descent...")
        
        # Pre-create a shuffled list of ratings for each epoch
        shuffled_data = list(ratings_data)                # -> Create a copy of ratings data that we can shuffle
        
        # Train over multiple epochs
        for epoch in range(self.n_epochs):                # -> Iterate through each training epoch
            # Shuffle the data each epoch
            random.shuffle(shuffled_data)                 # -> Randomly reorder the training examples
            
            squared_error = 0                             # -> Track total squared error for this epoch
            
            for user_id, item_id, rating in shuffled_data: # -> Iterate through each rating
                # Skip if user or item not in our mappings
                if user_id not in self.user_id_to_index or item_id not in self.item_id_to_index:
                    continue
                    
                # Get indices
                user_idx = self.user_id_to_index[user_id]  # -> Convert user_id to matrix index
                item_idx = self.item_id_to_index[item_id]  # -> Convert item_id to matrix index
                
                # Get implicit feedback factor - sum of implicit item factors for items user has rated
                implicit_sum = np.zeros(self.n_factors)    # -> Initialize array of zeros for summation
                sqrt_count = self.sqrt_user_rated_counts[user_id] # -> Get normalization factor 1/sqrt(|I_u|)
                
                if sqrt_count > 0:
                    # For each item the user has rated, add its implicit factor to the sum
                    for implicit_item_idx in self.user_rated_items[user_id]:
                        # Add the item's implicit factors to our sum
                        implicit_sum += self.item_implicit_factors[implicit_item_idx]
                    
                    # Normalize by sqrt(number of items rated)
                    implicit_sum *= sqrt_count            # -> Multiply by 1/sqrt(|I_u|) for normalization
                
                # Compute prediction with implicit feedback component
                # prediction = global_mean + user_bias + item_bias + 
                #              (user_factors + implicit_weight * implicit_sum)·item_factors
                # This is the key difference between SVD and SVD++
                pred = (
                    self.global_mean                        # -> Start with global average rating
                    + self.user_biases[user_id]             # -> Add user bias
                    + self.item_biases[item_id]             # -> Add item bias
                    + np.dot(
                        self.user_factors[user_idx] + self.implicit_weight * implicit_sum,  # -> User vector with implicit feedback
                        self.item_factors[item_idx]        # -> Item vector
                    )
                )
                
                # Calculate error
                error = rating - pred                       # -> Difference between actual and predicted rating
                squared_error += error ** 2                 # -> Track total squared error for RMSE calculation
                
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
                
                # Update item factors - note that we use the combined user vector (explicit + implicit)
                self.item_factors[item_idx] += self.lr * (
                    error * (old_user_factors + self.implicit_weight * implicit_sum) 
                    - self.reg * old_item_factors
                )
                
                # Update implicit item factors (for all items user has rated)
                if sqrt_count > 0:
                    # Gradient for implicit factors is more complex - need to update for each item user has rated
                    for implicit_item_idx in self.user_rated_items[user_id]:
                        self.item_implicit_factors[implicit_item_idx] += self.lr * (
                            error * self.implicit_weight * sqrt_count * old_item_factors
                            - self.reg * self.item_implicit_factors[implicit_item_idx]
                        )
            
            # Print epoch progress
            rmse = math.sqrt(squared_error / len(shuffled_data))  # -> Calculate Root Mean Square Error
            print(f"Epoch {epoch+1}/{self.n_epochs}: RMSE = {rmse:.4f}")
            
            # Adaptive learning rate - reduce over time
            self.lr *= 0.9                                # -> Decrease learning rate by 10% each epoch
        
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
        """
        Example:
        For a prediction of User 101 rating Movie 203:
        
        # User 101 has rated Movies 201, 202, and 204
        # Movie 203 hasn't been rated by User 101
        
        self.global_mean = 3.67
        self.user_biases[101] = 0.33
        self.item_biases[203] = 0.0
        
        # Explicit factors
        self.user_factors[user_idx] = [0.5, 0.8]
        self.item_factors[item_idx] = [0.8, 0.1]
        
        # Implicit factors for all items rated by User 101
        self.item_implicit_factors[0] = [0.1, 0.2]  # Movie 201
        self.item_implicit_factors[1] = [0.3, 0.1]  # Movie 202
        self.item_implicit_factors[3] = [0.1, 0.4]  # Movie 204
        
        # Calculate implicit sum
        implicit_sum = ([0.1, 0.2] + [0.3, 0.1] + [0.1, 0.4]) * 0.577 = [0.5, 0.7] * 0.577 = [0.289, 0.404]
        
        # Apply implicit weight
        weighted_implicit_sum = [0.289, 0.404] * 0.1 = [0.0289, 0.0404]
        
        # Combined user vector
        combined_user_vector = [0.5, 0.8] + [0.0289, 0.0404] = [0.5289, 0.8404]
        
        # Final prediction
        prediction = 3.67 + 0.33 + 0.0 + dot([0.5289, 0.8404], [0.8, 0.1])
                  = 4.0 + (0.5289*0.8 + 0.8404*0.1)
                  = 4.0 + 0.42312 + 0.08404
                  = 4.5072
        
        Final prediction after clipping to [1, 5] range: 4.5072
        """
        # Check if user and item exist in the training data
        if user_id not in self.user_id_to_index or item_id not in self.item_id_to_index:
            # For cold start, return global mean
            return self.global_mean                        # -> Return global mean for unknown users/items
        
        # Get indices
        user_idx = self.user_id_to_index[user_id]          # -> Convert user_id to matrix index
        item_idx = self.item_id_to_index[item_id]          # -> Convert item_id to matrix index
        
        # Calculate implicit feedback component
        implicit_sum = np.zeros(self.n_factors)            # -> Initialize implicit sum vector with zeros
        sqrt_count = self.sqrt_user_rated_counts.get(user_id, 0)  # -> Get normalization factor
        
        if sqrt_count > 0:
            # Sum all implicit item factors for items this user has rated
            for implicit_item_idx in self.user_rated_items[user_id]:
                implicit_sum += self.item_implicit_factors[implicit_item_idx]
            
            # Normalize by sqrt(number of items rated)
            implicit_sum *= sqrt_count                     # -> Apply normalization
        
        # Compute prediction with implicit feedback
        prediction = (
            self.global_mean                               # -> Start with global average rating
            + self.user_biases[user_id]                    # -> Add user bias
            + self.item_biases[item_id]                    # -> Add item bias
            + np.dot(
                self.user_factors[user_idx] + self.implicit_weight * implicit_sum,  # -> Combined user vector
                self.item_factors[item_idx]                # -> Item vector
            )
        )
        
        # Clip to valid rating range
        prediction = max(1.0, min(5.0, prediction))        # -> Ensure rating is between 1 and 5
        
        return prediction                                  # -> Return the predicted rating
    
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
    
    """
    | Agorithms                       | RMSE   | MAE       |
    |---------------------------------|--------|-----------|
    | Random                          | 1.4385 |    1.1478 |
    | Collaborative Filtering         |--------|-----------|
    | SVD++                           | 0.9187 |    0.7112 |   
    """

    # Run evaluation
    print("Evaluating algorithms...")
    evaluator.Evaluate(False)
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)