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
        self.n_factors = n_factors                         # -> Number of latent factors (dimensionality of factor vectors)
        self.n_epochs = n_epochs                           # -> Number of training iterations over the dataset
        self.lr = lr                                       # -> Initial learning rate for SGD
        self.user_reg = user_reg                           # -> Regularization parameter for user factors (controls overfitting)
        self.item_reg = item_reg                           # -> Regularization parameter for item factors (controls overfitting)
        self.lr_decay = lr_decay                           # -> Learning rate decay factor (reduces lr each epoch)
        self.min_lr = min_lr                               # -> Minimum learning rate to prevent too small steps
        self.init_std = init_std                           # -> Standard deviation for random initialization
        
        # These will be initialized during fitting
        self.user_factors = None                           # -> Will store the user latent factor matrix (users × factors)
        self.item_factors = None                           # -> Will store the item latent factor matrix (items × factors)
        self.global_mean = 0                               # -> Global average of all ratings
        
        # Mappings for user and item IDs
        self.user_id_to_index = {}                         # -> Dictionary of user_id -> index
        self.item_id_to_index = {}                         # -> Dictionary of item_id -> index
        self.user_index_to_id = {}                         # -> Dictionary of index -> user_id
        self.item_index_to_id = {}                         # -> Dictionary of index -> item_id
        
        # Rating normalization parameters
        self.min_rating = 1.0                              # -> Minimum rating in the dataset
        self.max_rating = 5.0                              # -> Maximum rating in the dataset
        self.rating_range = 4.0                            # -> Range of ratings (max - min)
    
    def fit(self, ratings_data):
        """
        Train the PMF model using SGD
        
        Args:
            ratings_data: List of (user_id, item_id, rating) tuples
        """
        """
        Example of PMF with a mini dataset:
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

        # First, we process the data to create our mappings
        self.users = [101, 102, 103]
        self.items = [201, 202, 203, 204, 205]
        self.user_id_to_index = {101: 0, 102: 1, 103: 2}
        self.item_id_to_index = {201: 0, 202: 1, 203: 2, 204: 3, 205: 4}
        self.user_index_to_id = {0: 101, 1: 102, 2: 103}
        self.item_index_to_id = {0: 201, 1: 202, 2: 203, 3: 204, 4: 205}
        
        # Calculate the global mean and determine rating range
        self.global_mean = (5 + 3 + 4 + 4 + 5 + 3 + 4 + 2 + 3) / 9 = 3.67
        self.min_rating = 2.0  # Minimum rating in our dataset
        self.max_rating = 5.0  # Maximum rating in our dataset
        self.rating_range = 5.0 - 2.0 = 3.0

        # Initialize parameters
        n_users = 3
        n_items = 5
        
        # Now we normalize the ratings to [0, 1] scale for better numerical stability
        normalized_ratings_data = [
            (101, 201, (5-2)/3 = 1.00),  # User 101 rated Movie 201 with 1.00 (normalized)
            (101, 202, (3-2)/3 = 0.33),  # User 101 rated Movie 202 with 0.33 (normalized)
            (101, 204, (4-2)/3 = 0.67),  # User 101 rated Movie 204 with 0.67 (normalized)
            (102, 201, (4-2)/3 = 0.67),  # User 102 rated Movie 201 with 0.67 (normalized)
            (102, 203, (5-2)/3 = 1.00),  # User 102 rated Movie 203 with 1.00 (normalized)
            (102, 205, (3-2)/3 = 0.33),  # User 102 rated Movie 205 with 0.33 (normalized)
            (103, 202, (4-2)/3 = 0.67),  # User 103 rated Movie 202 with 0.67 (normalized)
            (103, 203, (2-2)/3 = 0.00),  # User 103 rated Movie 203 with 0.00 (normalized)
            (103, 204, (3-2)/3 = 0.33)   # User 103 rated Movie 204 with 0.33 (normalized)
        ]
        
        user_momentum = [ 0.0, 0.0, 0.0 ]            # Momentum for user factors
        item_momentum = [ 0.0, 0.0, 0.0, 0.0, 0.0 ]  # Momentum for item factors

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

        ratings_dict = {
            101: {201: 1.00, 202: 0.33, 204: 0.67},  # User 101 ratings (normalized)
            102: {201: 0.67, 203: 1.00, 205: 0.33},  # User 102 ratings (normalized)
            103: {202: 0.67, 203: 0.00, 204: 0.33}   # User 103 ratings (normalized)
        }

        # Key difference in PMF: We initialize factors from a Gaussian distribution
        # User factors matrix (users × factors)
        self.user_factors = np.random.normal(0, 0.1, (3, 2))   # Example with 2 factors
        # Example user factors:
        self.user_factors = np.array([
            [0.05, 0.12],  # User 101
            [0.08, -0.03],  # User 102
            [-0.04, 0.09]   # User 103
        ])

        # Item factors matrix (items × factors)
        self.item_factors = np.random.normal(0, 0.1, (5, 2))   # Example with 2 factors
        # Example item factors:
        self.item_factors = np.array([
            [0.15, -0.02],  # Movie 201
            [0.09, 0.08],  # Movie 202
            [0.02, 0.13],  # Movie 203
            [0.11, 0.07],  # Movie 204
            [-0.05, 0.10]   # Movie 205
        ])
        
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

            pred = [0.08, -0.03] * [-0.05, 0.10] 
                 = 0.08 * -0.05 + -0.03 * 0.10 
                 = -0.0041

            pred = max(0, min(1, -0.0041)) = 0

            error = (3 - 0) = 3.0
            squared_error += 3.0 ** 2 = 9.0

            old_user_factors = [0.08, -0.03]  # User 102 factors before update
            old_item_factors = [-0.05, 0.10]  # Movie 205 factors before update
            
            # Parameters
            user_reg=0.1 
            item_reg=0.1

            user_gradient = -3.0 * [-0.05, 0.10] + 0.1 * [0.08, -0.03] 
                          = [-0.15, 0.30] + [0.008, -0.003]
                          = [-0.14199999999999999, 0.297]
            item_gradient = -3.0 * [0.08, -0.03] + 0.1 * [-0.05, 0.10]
                          = [-0.24, 0.09] + [-0.005, 0.01]
                          = [-0.245, 0.1] 

            user_momentum = 0.9 * [0.0, 0.0] - 0.005 * [-0.14199999999999999, 0.297]
                          = [0.0007099999999999999, -0.001485]
            item_momentum = 0.9 * [0.0, 0.0] - 0.005 * [-0.245, 0.1]
                          = [0.001225, -0.0005]  

            self.user_factors[1] += [0.0007099999999999999, -0.001485]
            self.item_factors[4] += [0.001225, -0.0005]

        # Example of user_factors and item_factors after training:
        self.user_factors = np.array([
            [0.05, 0.12],   # User 101
            [0.08, -0.03],  # User 102
            [-0.04, 0.09]   # User 103
        ])
        self.item_factors = np.array([
            [0.15, -0.02],  # Movie 201
            [0.09, 0.08],   # Movie 202
            [0.02, 0.13],   # Movie 203
            [0.11, 0.07],   # Movie 204
            [-0.05, 0.10]   # Movie 205
        ])

        # Explanation of PMF:
        # The ratings matrix R would be:

        R = [
            [5, 3, ?, 4, ?],  # User 101
            [4, ?, 5, ?, 3],  # User 102
            [?, 4, 2, 3, ?]   # User 103
        ]

        # PMF approximates this as the product of three matrices: R ≈ U·V^T

        Where:
            - U is the user factors matrix (users × factors)
                U = [
                    [0.05, 0.12],   # User 101 factors
                    [0.08, -0.03],  # User 102 factors
                    [-0.04, 0.09]   # User 103 factors
                ]
            - V^T is the transposed item factors matrix (factors × items)
                V = [
                    [0.15, -0.02],  # Movie 201 factors
                    [0.09, 0.08],   # Movie 202 factors
                    [0.02, 0.13],   # Movie 203 factors
                    [0.11, 0.07],   # Movie 204 factors
                    [-0.05, 0.10]   # Movie 205 factors
                ]
                V^T = [
                    [0.15, 0.09, 0.02, 0.11, -0.05],
                    [-0.02, 0.08, 0.13, 0.07, 0.10]
                ]
        """

        print("Processing ratings data for PMF matrix factorization...")
        
        # Extract all users, items, and ratings
        users = set()                                   # -> Create a set to track unique user IDs
        items = set()                                   # -> Create a set to track unique item IDs
        ratings_list = []                               # -> Create a list to store all rating values
        
        for user_id, item_id, rating in ratings_data:   # -> Iterate through each rating tuple
            users.add(user_id)                          # -> Add user_id to the users set
            items.add(item_id)                          # -> Add item_id to the items set
            ratings_list.append(rating)                 # -> Append rating to the ratings list
        
        # Create mappings for users and items
        self.users = sorted(list(users))                # -> Convert users set to a sorted list
        self.items = sorted(list(items))                # -> Convert items set to a sorted list
        
        # Create dictionary mappings between IDs and indices
        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(self.users)}    # -> Create mapping from user_id to index
        self.item_id_to_index = {item_id: idx for idx, item_id in enumerate(self.items)}    # -> Create mapping from item_id to index
        
        self.user_index_to_id = {idx: user_id for user_id, idx in self.user_id_to_index.items()} # -> Create mapping from index to user_id
        self.item_index_to_id = {idx: item_id for item_id, idx in self.item_id_to_index.items()} # -> Create mapping from index to item_id
        
        # Calculate global mean and rating bounds
        self.global_mean = sum(ratings_list) / len(ratings_list) if ratings_list else 0   # -> Calculate average of all ratings
        self.min_rating = min(ratings_list)             # -> Find minimum rating value in dataset
        self.max_rating = max(ratings_list)             # -> Find maximum rating value in dataset
        self.rating_range = self.max_rating - self.min_rating  # -> Calculate range of ratings
        
        # Initialize parameters
        n_users = len(self.users)                       # -> Number of unique users
        n_items = len(self.items)                       # -> Number of unique items
        
        # Initialize factor matrices with random values from a Gaussian distribution
        # This follows the probabilistic nature of PMF where factors have Gaussian priors
        self.user_factors = np.random.normal(0, self.init_std, (n_users, self.n_factors))   # -> Initialize user factor matrix with Gaussian noise
        self.item_factors = np.random.normal(0, self.init_std, (n_items, self.n_factors))   # -> Initialize item factor matrix with Gaussian noise
        
        # Normalize ratings to [0, 1] for better numerical stability
        def normalize_rating(r):                        # -> Helper function to normalize ratings to [0,1] range
            return (r - self.min_rating) / self.rating_range
        
        # Convert to normalized ratings
        normalized_ratings_data = [(u, i, normalize_rating(r)) for u, i, r in ratings_data]  # -> Create normalized version of all ratings
        
        # Train using SGD with momentum
        print("Training PMF model using Stochastic Gradient Descent...")
        
        current_lr = self.lr                            # -> Start with initial learning rate
        momentum = 0.9                                  # -> Momentum coefficient for faster convergence
        
        # User and item factor momentum terms
        user_momentum = np.zeros_like(self.user_factors)  # -> Initialize momentum for user factors to zeros
        item_momentum = np.zeros_like(self.item_factors)  # -> Initialize momentum for item factors to zeros
        
        # Pre-create a shuffled list of ratings for each epoch
        shuffled_data = list(normalized_ratings_data)    # -> Create a copy of normalized ratings data that we can shuffle
        
        # Create dictionary for faster lookup during training
        ratings_dict = {}                                # -> Dictionary to store user->item->rating mappings
        for user_id, item_id, rating in normalized_ratings_data:  # -> Iterate through normalized ratings
            if user_id not in ratings_dict:
                ratings_dict[user_id] = {}               # -> Initialize empty dict for this user if needed
            ratings_dict[user_id][item_id] = rating     # -> Store the normalized rating
        
        # Train over multiple epochs
        for epoch in range(self.n_epochs):              # -> Iterate through each training epoch
            # Shuffle the data each epoch
            random.shuffle(shuffled_data)               # -> Randomly reorder the training examples
            
            squared_error = 0                           # -> Track total squared error for this epoch
            
            for user_id, item_id, norm_rating in shuffled_data:  # -> Iterate through each normalized rating
                # Skip if user or item not in our mappings
                if user_id not in self.user_id_to_index or item_id not in self.item_id_to_index:
                    continue
                    
                # Get indices
                user_idx = self.user_id_to_index[user_id]  # -> Convert user_id to matrix index
                item_idx = self.item_id_to_index[item_id]  # -> Convert item_id to matrix index
                
                # Compute prediction (normalized)
                # In PMF, we only use dot product without biases
                pred = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])  # -> Calculate predicted rating
                
                # Clip prediction to [0, 1] range
                pred = max(0, min(1, pred))             # -> Ensure prediction is in normalized range
                
                # Calculate prediction error
                error = norm_rating - pred              # -> Difference between actual and predicted rating
                squared_error += error ** 2             # -> Track squared error for RMSE
                
                # Save old factors for update calculations
                old_user_factors = self.user_factors[user_idx].copy()  # -> Copy current user factors
                old_item_factors = self.item_factors[item_idx].copy()  # -> Copy current item factors
                
                # Calculate gradients with regularization
                # Note separate regularization terms for users and items
                user_gradient = -error * old_item_factors + self.user_reg * old_user_factors  # -> Calculate gradient for user factors
                item_gradient = -error * old_user_factors + self.item_reg * old_item_factors  # -> Calculate gradient for item factors
                
                # Update with momentum
                user_momentum[user_idx] = momentum * user_momentum[user_idx] - current_lr * user_gradient  # -> Update user momentum
                item_momentum[item_idx] = momentum * item_momentum[item_idx] - current_lr * item_gradient  # -> Update item momentum
                
                # Apply updates
                self.user_factors[user_idx] += user_momentum[user_idx]  # -> Update user factors with momentum
                self.item_factors[item_idx] += item_momentum[item_idx]  # -> Update item factors with momentum
            
            # Print epoch progress - converting RMSE back to original scale
            rmse = math.sqrt(squared_error / len(shuffled_data)) * self.rating_range  # -> Calculate RMSE in original rating scale
            print(f"Epoch {epoch+1}/{self.n_epochs}: RMSE = {rmse:.4f}, Learning Rate = {current_lr:.6f}")
            
            # Decay learning rate
            current_lr = max(self.min_lr, current_lr * self.lr_decay)  # -> Reduce learning rate but not below minimum
            
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

        """
        Example:
            For User 101 and Movie 203:
            
            self.user_factors[user_idx] = [0.05, 0.12]  # Trained factors for User 101
            self.item_factors[item_idx] = [0.02, 0.13]  # Trained factors for Movie 203
            
            # In PMF, we normalize to [0,1] range and denormalize at the end
            
            # Compute normalized prediction
            norm_pred = dot([0.05, 0.12], [0.02, 0.13])
                     = 0.05*0.02 + 0.12*0.13
                     = 0.001 + 0.0156
                     = 0.0166
            
            # Clip to [0, 1] range
            norm_pred = max(0, min(1, 0.0166)) = 0.0166
            
            # Convert back to original scale
            # min_rating = 2, max_rating = 5, rating_range = 3
            prediction = 0.0166 * 3 + 2
                       = 0.0498 + 2
                       = 2.0498
            
            # Clip to valid range [2, 5]
            prediction = max(2, min(5, 2.0498)) = 2.0498
        """
        # Check if user and item exist in the training data
        if user_id not in self.user_id_to_index or item_id not in self.item_id_to_index:
            # For cold start, return global mean
            return self.global_mean                     # -> Return global mean for unknown users/items
        
        # Get indices
        user_idx = self.user_id_to_index[user_id]       # -> Convert user_id to matrix index
        item_idx = self.item_id_to_index[item_id]       # -> Convert item_id to matrix index
        
        # Compute normalized prediction
        norm_pred = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])  # -> Calculate dot product of user and item factors
        
        # Clip normalized prediction to [0, 1]
        norm_pred = max(0, min(1, norm_pred))           # -> Ensure prediction is in normalized range
        
        # Convert back to original rating scale
        prediction = norm_pred * self.rating_range + self.min_rating  # -> Convert from normalized [0,1] back to original rating scale
        
        # Clip to valid rating range
        prediction = max(self.min_rating, min(self.max_rating, prediction))  # -> Ensure rating is within valid range
        
        return prediction                               # -> Return the predicted rating
    
    def get_user_factors(self, user_id):
        """Get latent factors for a specific user"""
        if user_id in self.user_id_to_index:
            return self.user_factors[self.user_id_to_index[user_id]]  # -> Return the latent factor vector for this user
        return None
    
    def get_item_factors(self, item_id):
        """Get latent factors for a specific item"""
        if item_id in self.item_id_to_index:
            return self.item_factors[self.item_id_to_index[item_id]]  # -> Return the latent factor vector for this item
        return None
    
    def calculate_posterior_variance(self):
        """
        Calculate the posterior variance of user and item factors
        This is specific to PMF's probabilistic interpretation
        """
        # For PMF, posterior variance is related to the regularization parameters
        user_variance = 1.0 / self.user_reg if self.user_reg > 0 else float('inf')  # -> Calculate variance for user factors
        item_variance = 1.0 / self.item_reg if self.item_reg > 0 else float('inf')  # -> Calculate variance for item factors
        
        return {
            'user_variance': user_variance,              # -> Return user factor variance
            'item_variance': item_variance               # -> Return item factor variance
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
    
    """
    | Agorithms                       | RMSE   | MAE       |
    |---------------------------------|--------|-----------|
    | Random                          | 1.4385 |    1.1478 |
    | Collaborative Filtering         |--------|-----------|
    | PMF                             | 1.0549 |    0.8546 |   
    """
    
    # Run evaluation
    print("Evaluating algorithms...")
    evaluator.Evaluate(False)
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)