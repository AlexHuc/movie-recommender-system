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
import numpy as np
import math
import random

# Libs used for AdaptedNMF
from surprise import AlgoBase, PredictionImpossible

# Libs to save and load models
from datetime import datetime
import pickle

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
        self.n_factors = n_factors                      # -> Number of latent factors (dimensionality of user/item vectors)
        self.n_epochs = n_epochs                        # -> Number of training iterations over the entire dataset
        self.lr = lr                                    # -> Learning rate for gradient descent (controls step size)
        self.reg = reg                                  # -> Regularization parameter (prevents overfitting)
        self.beta = beta                                # -> Parameter for enforcing non-negativity constraint
        
        # These will be initialized during fitting
        self.user_factors = None                        # -> Will store the user latent factor matrix (users × factors)
        self.item_factors = None                        # -> Will store the item latent factor matrix (items × factors)
        self.global_mean = 0                            # -> Global average of all ratings
        
        # Mappings for user and item IDs
        self.user_id_to_index = {}                      # -> Dictionary of user_id -> index
        self.item_id_to_index = {}                      # -> Dictionary of item_id -> index
        self.user_index_to_id = {}                      # -> Dictionary of index -> user_id
        self.item_index_to_id = {}                      # -> Dictionary of index -> item_id
    
    def fit(self, ratings_data):
        """
        Train the NMF model using gradient descent
        
        Args:
            ratings_data: List of (user_id, item_id, rating) tuples
        """
        """
        Example of the NMF model with a mini dataset:
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

        # First, we process the data to create our mappings:
        self.users = [101, 102, 103]
        self.items = [201, 202, 203, 204, 205]
        self.user_id_to_index = {101: 0, 102: 1, 103: 2}
        self.item_id_to_index = {201: 0, 202: 1, 203: 2, 204: 3, 205: 4}
        self.user_index_to_id = {0: 101, 1: 102, 2: 103}
        self.item_index_to_id = {0: 201, 1: 202, 2: 203, 3: 204, 4: 205}
        
        # Calculate global mean rating
        self.global_mean = (5 + 3 + 4 + 4 + 5 + 3 + 4 + 2 + 3) / 9 = 3.67
        
        n_users = 3  # Number of unique users
        n_items = 5  # Number of unique items
        
        # Key difference in NMF: Initialize factor matrices with POSITIVE random values
        # unlike SVD which uses normal distribution that can generate negative values
        self.user_factors = np.random.uniform(0.01, 0.1, (n_users, self.n_factors))
        # Example user factors (non-negative):
        self.user_factors = np.array([
            [0.05, 0.08],  # User 101 - all values are positive
            [0.09, 0.02],  # User 102 - all values are positive
            [0.03, 0.07]   # User 103 - all values are positive
        ])

        self.item_factors = np.random.uniform(0.01, 0.1, (n_items, self.n_factors))
        # Example item factors (non-negative):
        self.item_factors = np.array([
            [0.06, 0.03],  # Movie 201 - all values are positive
            [0.02, 0.08],  # Movie 202 - all values are positive
            [0.09, 0.01],  # Movie 203 - all values are positive
            [0.04, 0.09],  # Movie 204 - all values are positive
            [0.05, 0.06]   # Movie 205 - all values are positive
        ])

        ratings_dict = {
            101: {201: 1.00, 202: 0.33, 204: 0.67},  # User 101 ratings (normalized)
            102: {201: 0.67, 203: 1.00, 205: 0.33},  # User 102 ratings (normalized)
            103: {202: 0.67, 203: 0.00, 204: 0.33}   # User 103 ratings (normalized)
        }
        
        # Normalize ratings to [0, 1] scale for better numerical stability
        min_rating = min(ratings_list) = 2
        max_rating = max(ratings_list) = 5
        rating_range = max_rating - min_rating = 3
        
        # Create normalized ratings dictionary
        normalized_ratings = {
            (101, 201): (5-2)/3 = 1.0,
            (101, 202): (3-2)/3 = 0.33,
            (101, 204): (4-2)/3 = 0.67,
            (102, 201): (4-2)/3 = 0.67,
            (102, 203): (5-2)/3 = 1.0,
            (102, 205): (3-2)/3 = 0.33,
            (103, 202): (4-2)/3 = 0.67,
            (103, 203): (2-2)/3 = 0.0,
            (103, 204): (3-2)/3 = 0.33
        }

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

            # First iteration in the suffled data:
            # User 102 rated Movie 205 with 3 stars

            user_idx = 1 # User 102 index in user_id_to_index
            item_idx = 4 # Movie 205 index in item_id_to_index

            norm_rating = normalized_ratings[(102, 205)]  
                        = 0.33  # Normalized rating for User 102 and Movie 205

            pred = np.dot(self.user_factors[1], self.item_factors[4]
                 = [0.09, 0.02] * [0.05, 0.06]
                 = 0.09 * 0.05 + 0.02 * 0.06
                 = 0.0045 + 0.0012 
                 = 0.0057

            error = norm_rating - pred = 0.33 - 0.0057 = 0.3243
            squared_error += error ** 2 = 0.3243 ** 2 = 0.1051

            old_user_factors = [0.09, 0.02]
            old_item_factors = [0.05, 0.06]

            # Calculatint U
            for i in range(0, self.n_factors = 15):
                # First iteration (i=0):
                update = 0.01 * (0.3243 * 0.05 - 0.02 * 0.09) = 0.0001
                self.user_factors[user_idx, i] += update
                                               = [0.09 + 0.0001, 0.02 + 0.0001]
                self.user_factors[user_idx, i] = max(0, self.user_factors[user_idx, i]) 
                                               = [0.0901, 0.0201]

                 # Second iteration (i=1):   
                update = 0.01 * (0.3243 * 0.06 - 0.02 * 0.02) = 0.0002
                self.user_factors[user_idx, i] += update
                                               = [0.0901, 0.0201 + 0.0002]
                self.user_factors[user_idx, i] = max(0, self.user_factors[user_idx, i])
                                               = [0.0901, 0.0203]

                ...
            
            # Calculatint V
            for i in range(0, self.n_factors = 15):
                # First iteration (i=0):
                update = 0.01 * (0.3243 * 0.09 - 0.02 * 0.05) = 0.0002
                self.item_factors[item_idx, i] += update
                                               = [0.05 + 0.0002, 0.06 + 0.0002]
                self.item_factors[item_idx, i] = max(0, self.item_factors[item_idx, i])
                                               = [0.0502, 0.0602]

                 # Second iteration (i=1):   
                update = 0.01 * (0.3243 * 0.02 - 0.02 * 0.06) = -0.0001
                self.item_factors[item_idx, i] += update
                                               = [0.0502, 0.0602 - 0.0001]
                self.item_factors[item_idx, i] = max(0, self.item_factors[item_idx, i])
                                               = [0.0502, 0.0601]

                ...
        
        # Example of user_factors and item_factors after training:
        self.user_factors = np.array([
            [0.05058676, 0.08028238],  # User 101 factors after training
            [0.08020001, 0.03015601],  # User 102 factors after training
            [0.07032301, 0.04023205]   # User 103 factors after training
        ])
        self.item_factors = np.array([
            [0.06012345, 0.03067890],  # Movie 201 factors after training
            [0.02045678, 0.08012345],  # Movie 202 factors after training
            [0.09048711, 0.01029761],  # Movie 203 factors after training
            [0.04056789, 0.09067890],  # Movie 204 factors after training
            [0.05067890, 0.06012345]   # Movie 205 factors after training
        ])

         # Explanation of NMF:
         # The ratings matrix R would be:

        R = [
            [5, 3, ?, 4, ?],  # User 101
            [4, ?, 5, ?, 3],  # User 102
            [?, 4, 2, 3, ?]   # User 103
        ]

        # PMF approximates this as the product of three matrices: R ≈ U·V^T
        - P is the user factor matrix (users × factors)
            P = [
                [0.05058676, 0.08028238],  # User 101 factors
                [0.08020001, 0.03015601],  # User 102 factors
                [0.07032301, 0.04023205]   # User 103 factors
            ]
        - Q is the item factor matrix (items × factors) 
            Q = [
                [0.06012345, 0.03067890],  # Movie 201 factors
                [0.02045678, 0.08012345],  # Movie 202 factors
                [0.09048711, 0.01029761],  # Movie 203 factors
                [0.04056789, 0.09067890],  # Movie 204 factors
                [0.05067890, 0.06012345]   # Movie 205 factors
            ]
        """
        print("Processing ratings data for NMF matrix factorization...")
        
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
        # Key difference in NMF: Initialize with small POSITIVE random values for non-negativity
        self.user_factors = np.random.uniform(0.01, 0.1, (n_users, self.n_factors))  # -> Initialize user factors with small positive values
        
        # Item factors matrix (items × factors)
        # Also initialized with small positive random values
        self.item_factors = np.random.uniform(0.01, 0.1, (n_items, self.n_factors))  # -> Initialize item factors with small positive values
        
        # Create mapping of (user, item) -> rating for faster lookups
        ratings_dict = {(user_id, item_id): rating for user_id, item_id, rating in ratings_data}  # -> Create dictionary for O(1) rating lookups
        
        # Normalize ratings to [0, 1] for better NMF performance
        min_rating = min(ratings_list)                    # -> Find minimum rating value
        max_rating = max(ratings_list)                    # -> Find maximum rating value
        rating_range = max_rating - min_rating            # -> Calculate range of ratings
        
        if rating_range > 0:
            normalized_ratings = {k: (v - min_rating) / rating_range for k, v in ratings_dict.items()}  # -> Scale ratings to [0,1] range
        else:
            normalized_ratings = ratings_dict             # -> If all ratings are identical, skip normalization
            
        # Store rating scale information for later denormalization during prediction
        self.min_rating = min_rating                      # -> Store minimum rating for denormalization
        self.rating_range = rating_range                  # -> Store rating range for denormalization
        
        # Train using gradient descent
        print("Training NMF model using gradient descent...")
        
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
                
                # Get normalized rating
                norm_rating = normalized_ratings[(user_id, item_id)]  # -> Get the normalized rating in [0,1] range
                
                # Compute prediction (dot product of user and item factors)
                pred = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])  # -> Calculate predicted rating
                
                # Calculate error (normalized)
                error = norm_rating - pred                # -> Calculate error between actual and predicted normalized rating
                squared_error += error ** 2               # -> Track squared error for RMSE calculation
                
                # Save old factors for updates
                old_user_factors = self.user_factors[user_idx].copy()  # -> Copy current user factors
                old_item_factors = self.item_factors[item_idx].copy()  # -> Copy current item factors
                
                # Update user factors with gradient descent
                # Include non-negativity constraint through regularization
                for f in range(self.n_factors):           # -> Update each factor dimension separately
                    update = (
                        self.lr * (error * old_item_factors[f] - self.reg * old_user_factors[f])  # -> Calculate gradient update
                    )
                    
                    # Ensure non-negativity with soft constraint
                    if old_user_factors[f] + update < 0:  # -> Check if update would make factor negative
                        # Reduce negative updates to maintain non-negativity
                        update = -old_user_factors[f] * self.beta  # -> Apply softer negative update based on beta parameter
                        
                    self.user_factors[user_idx, f] += update  # -> Apply the update to user factor
                    
                    # Ensure strict non-negativity by clipping to zero as lower bound
                    self.user_factors[user_idx, f] = max(0, self.user_factors[user_idx, f])  # -> Enforce non-negativity
                
                # Update item factors with gradient descent
                # Include non-negativity constraint
                for f in range(self.n_factors):           # -> Update each item factor dimension
                    update = (
                        self.lr * (error * old_user_factors[f] - self.reg * old_item_factors[f])  # -> Calculate gradient update
                    )
                    
                    # Ensure non-negativity with soft constraint
                    if old_item_factors[f] + update < 0:  # -> Check if update would make factor negative
                        # Reduce negative updates
                        update = -old_item_factors[f] * self.beta  # -> Apply softer negative update
                        
                    self.item_factors[item_idx, f] += update  # -> Apply the update to item factor
                    
                    # Ensure strict non-negativity
                    self.item_factors[item_idx, f] = max(0, self.item_factors[item_idx, f])  # -> Enforce non-negativity
            
            # Print epoch progress
            rmse = math.sqrt(squared_error / len(shuffled_data))  # -> Calculate normalized RMSE
            # Convert to original scale for better interpretability
            denorm_rmse = rmse * rating_range             # -> Scale RMSE back to original rating scale
            print(f"Epoch {epoch+1}/{self.n_epochs}: RMSE = {denorm_rmse:.4f}")
            
            # Adaptive learning rate - reduce over time
            self.lr *= 0.95                              # -> Decrease learning rate by 5% each epoch
        
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
        """
        Example:
            For User 101 and Movie 203:
            
            self.user_factors[user_idx] = [0.05058676, 0.08028238]  # Non-negative user factors
            self.item_factors[item_idx] = [0.09048711, 0.01029761]  # Non-negative item factors
            
            # Calculate normalized prediction (dot product)
            norm_pred = dot([0.05058676, 0.08028238], [0.09048711, 0.01029761])
            norm_pred = 0.05058676 * 0.09048711 + 0.08028238 * 0.01029761
            norm_pred = 0.00457717 + 0.00082673
            norm_pred = 0.0054039
            
            # Clip normalized prediction to [0, 1]
            norm_pred = max(0, min(1, 0.0054039)) = 0.0054039
            
            # Convert back to original rating scale
            # min_rating = 2, rating_range = 3
            prediction = 0.0054039 * 3 + 2 = 0.0162117 + 2 = 2.0162117
            
            # Clip to valid rating range [1, 5]
            prediction = max(1.0, min(5.0, 2.0162117)) = 2.0162117
        """
        # Check if user and item exist in the training data
        if user_id not in self.user_id_to_index or item_id not in self.item_id_to_index:
            # For cold start, return global mean
            return self.global_mean                       # -> Return global mean for unknown users/items
        
        # Get indices
        user_idx = self.user_id_to_index[user_id]        # -> Convert user_id to matrix index
        item_idx = self.item_id_to_index[item_id]        # -> Convert item_id to matrix index
        
        # Compute normalized prediction (dot product of user and item factors)
        norm_pred = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])  # -> Calculate dot product
        
        # Clip normalized prediction to [0, 1]
        norm_pred = max(0, min(1, norm_pred))            # -> Ensure prediction is in normalized range
        
        # Convert back to original rating scale
        prediction = norm_pred * self.rating_range + self.min_rating  # -> Convert from normalized [0,1] to original rating scale
        
        # Clip to valid rating range (typically 1-5)
        prediction = max(1.0, min(5.0, prediction))      # -> Ensure rating is within valid range
        
        return prediction                                # -> Return the predicted rating
    
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
            filename = f"../models/2_MatrixFactorization/adapted_nmf_model_{timestamp}.pkl"
        
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
    nmf_recommender = AdaptedNMF(n_factors=15, n_epochs=50, lr=0.01, reg=0.02, beta=0.02)
    evaluator.AddAlgorithm(nmf_recommender, "NMF")
    
    random_predictor = NormalPredictor()
    evaluator.AddAlgorithm(random_predictor, "Random")
    
    """
    | Agorithms                       | RMSE   | MAE       |
    |---------------------------------|--------|-----------|
    | Random                          | 1.4385 |    1.1478 |
    | Collaborative Filtering         |--------|-----------|
    | NMF                             | 1.0133 |    0.7930 |
    """

    # Run evaluation
    print("Evaluating algorithms...")
    evaluator.Evaluate(False)

    nmf_recommender.save_model()
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)