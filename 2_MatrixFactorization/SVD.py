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
            n_epochs (int):  Number of iterations for SGD
            lr (float):      Learning rate for SGD
            reg (float):     Regularization term for SGD
        """
        self.n_factors = n_factors                      # -> Number of latent factors to use in factorization (dimensionality of user/item vectors)
        self.n_epochs = n_epochs                        # -> Number of training iterations over the entire dataset
        self.lr = lr                                    # -> Learning rate for gradient descent (controls step size)
        self.reg = reg                                  # -> Regularization parameter (prevents overfitting)
        
        # These will be initialized during fitting
        self.user_factors = None                        # -> Will store the user latent factor matrix (users × factors)
        self.item_factors = None                        # -> Will store the item latent factor matrix (items × factors)
        self.global_mean = 0                            # -> Global average of all ratings
        self.user_biases = {}                           # -> User bias terms (user_id -> bias)
        self.item_biases = {}                           # -> Item bias terms (item_id -> bias)
        
        # Mappings for user and item IDs
        self.user_id_to_index = {}                      # -> Dictionary of user_id -> index
        self.item_id_to_index = {}                      # -> Dictionary of item_id -> index
        self.user_index_to_id = {}                      # -> Dictionary of index -> user_id
        self.item_index_to_id = {}                      # -> Dictionary of index -> item_id
    
    def fit(self, ratings_data):
        """
        Train the SVD model using SGD
        
        Args:
            ratings_data: List of (user_id, item_id, rating) tuples
        """

        """
        Example of the SVD model with a mini dataset:
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

        self.user_biases = {101: 0.33, 102: 0.0, 103: -0.33}                      # Example user biases
        self.item_biases = {201: 0.33, 202: 0.0, 203: 0.0, 204: 0.0, 205: -0.33}  # Example item biases

        self.user_counts = {101: 3, 102: 3, 103: 3}                               # Count of ratings per user
        self.item_counts = {201: 2, 202: 2, 203: 2, 204: 2, 205: 1}               # Count of ratings per item
        
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
                prediction = global_mean + user_bias + item_bias + dot(user_factors[user_idx], item_factors[item_idx]) = 
                            = 3.67 + 0.0 + -0.33 + [0.9, 0.2] * [0.2, 0.8] = 
                            = 3.67 + 0.0 + -0.33 + (0.9 * 0.2 + 0.2 * 0.8) =
                            = 3.67 + 0.0 + -0.33 + (0.18 + 0.16) =
                            = 3.67 + 0.0 + -0.33 + 0.34 
                            = 3.68

                error = rating - prediction = 5 - 3.68 = 1.32
                squared_error += error ** 2 = 1.32 ** 2 = 1.7424

                user_biases[user_id] += lr * (error - reg * user_biases[user_id]) = 
                                     = 0.0 + 0.005 * (1.32 - 0.02 * 0.0) =
                                     = 0.0 + 0.005 * 1.32 =
                                     = 0.0 + 0.0066 
                                     = 0.0066

                item_biases[item_id] += lr * (error - reg * item_biases[item_id]) =
                                     = -0.33 + 0.005 * (1.32 - 0.02 * -0.33) =
                                     = -0.33 + 0.005 * (1.32 + 0.0066) =
                                     = -0.33 + 0.005 * 1.3266 =
                                     = -0.33 + 0.006633 =
                                     = -0.323366

                user_factors[user_idx] += lr * (error * item_factors[item_idx] - reg * user_factors[user_idx]) =
                                       = [0.9, 0.2] + 0.005 * (1.32 * [0.2, 0.8] - 0.02 * [0.9, 0.2]) =
                                       = [0.9, 0.2] + 0.005 * ([0.264, 1.056] - [0.018, 0.004]) =
                                       = [0.9, 0.2] + 0.005 * ([0.246, 1.052]) =
                                       = [0.9, 0.2] + [0.00123, 0.00526] =
                                       = [0.90123, 0.20526]

                item_factors[item_idx] += lr * (error * user_factors[user_idx] - reg * item_factors[item_idx]) =
                                        = [0.2, 0.8] + 0.005 * (1.32 * [0.90123, 0.20526] - 0.02 * [0.2, 0.8]) =
                                        = [0.2, 0.8] + 0.005 * ([1.18962, 0.27192] - [0.004, 0.016]) =
                                        = [0.2, 0.8] + 0.005 * ([1.18562, 0.25592]) =
                                        = [0.2, 0.8] + [0.0059281, 0.0012796] =
                                        = [0.2059281, 0.8012796]     

        On the current example the user_factors and item_factors matrices would look like this in the final:
        # User factors matrix (users × factors)
        self.user_factors = np.array([    
            [0.90123, 0.20526],  # User 101
            [0.90123, 0.20526],  # User 102
            [0.30123, 0.70526]   # User 103
        ])
        # Item factors matrix (items × factors)
        self.item_factors = np.array([
            [0.2059281, 0.8012796],  # Movie 201
            [0.2029281, 0.8022796],  # Movie 202
            [0.8029281, 0.1012796],  # Movie 203
            [0.4049281, 0.9012796],  # Movie 204
            [0.1029281, 0.5012796]   # Movie 205
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
        
        For our implementation, we are using a variant called Funk SVD where:
        
        R ≈ global_mean + user_biases + item_biases + U * V^T
        
        Where:
        - global_mean: The average rating across all users and items (e.g., 3.67)
        - user_biases: User-specific rating tendencies (e.g., {101: 0.33, 102: 0.0, 103: -0.33})
        - item_biases: Item-specific rating tendencies (e.g., {201: 0.33, 202: 0.0, 203: 0.0, 204: 0.0, 205: -0.33})
        - U: User factors matrix with n_factors columns (e.g., 3x2 for 3 users and 2 factors)
              [0.5, 0.8]
              [0.9, 0.2]
              [0.3, 0.7]
        - V: Item factors matrix with n_factors columns (e.g., 5x2 for 5 items and 2 factors)
              [0.6, 0.3]
              [0.2, 0.8]
              [0.8, 0.1]
              [0.4, 0.9]
              [0.1, 0.5]
        
        To predict a rating for User 101 on Movie 203:
        prediction = global_mean + user_bias_101 + item_bias_203 + dot(U[101], V[203])
                   = 3.67 + 0.33 + 0.0 + dot([0.5, 0.8], [0.8, 0.1])
                   = 4.0 + 0.5*0.8 + 0.8*0.1
                   = 4.0 + 0.4 + 0.08
                   = 4.48

        """

        print("Processing ratings data for SVD matrix factorization...")
        
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
        # Each row represents a user's latent factors, initialized with small random values
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))   # -> Initialize user factor matrix with random values
        
        # Item factors matrix (items × factors)
        # Each row represents an item's latent factors, initialized with small random values
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))   # -> Initialize item factor matrix with random values
        
        # Initialize biases to zero
        self.user_biases = {u: 0.0 for u in self.users}  # -> Initialize user biases to 0.0
        self.item_biases = {i: 0.0 for i in self.items}  # -> Initialize item biases to 0.0
        
        # Train using SGD (stochastic gradient descent)
        print("Training SVD model using Stochastic Gradient Descent...")
        
        # Keep track of the count of instances for each user and item (not actually used in the current implementation)
        user_counts = {u: 0 for u in self.users}         # -> Count how many ratings each user has made
        item_counts = {i: 0 for i in self.items}         # -> Count how many ratings each item has received
        
        for user_id, item_id, rating in ratings_data:     # -> Iterate through all ratings to count them
            user_counts[user_id] += 1                     # -> Increment count for this user
            item_counts[item_id] += 1                     # -> Increment count for this item
        
        # Pre-create a shuffled list of ratings for each epoch
        shuffled_data = list(ratings_data)                # -> Create a copy of ratings data that we can shuffle
        
        # Train over multiple epochs
        for epoch in range(self.n_epochs):                # -> Iterate through each training epoch
            # Shuffle the data each epoch
            random.shuffle(shuffled_data)                 # -> Randomly reorder the training examples
            
            squared_error = 0                             # -> Track total squared error for this epoch
            
            for user_id, item_id, rating in shuffled_data: # -> Iterate through each rating
                # Get indices
                user_idx = self.user_id_to_index[user_id]  # -> Convert user_id to matrix index
                item_idx = self.item_id_to_index[item_id]  # -> Convert item_id to matrix index
                
                # Compute prediction
                # prediction = global_mean + user_bias + item_bias + user_factors·item_factors
                # For example: prediction = 3.67 + 0.33 + 0.0 + dot([0.5, 0.8], [0.8, 0.1]) = 4.48
                pred = (
                    self.global_mean                        # -> Start with global average rating
                    + self.user_biases[user_id]             # -> Add user bias
                    + self.item_biases[item_id]             # -> Add item bias
                    + np.dot(self.user_factors[user_idx], self.item_factors[item_idx]) # -> Add dot product of user and item factors
                )
                
                # Calculate error
                error = rating - pred                       # -> Difference between actual and predicted rating
                squared_error += error ** 2                 # -> Track total squared error for RMSE calculation
                
                # Update biases
                # For example:
                # If error = 0.5, lr = 0.005, reg = 0.02, user_bias = 0.33:
                # new_user_bias = 0.33 + 0.005 * (0.5 - 0.02 * 0.33) = 0.33 + 0.005 * 0.4934 = 0.332467
                self.user_biases[user_id] += self.lr * (error - self.reg * self.user_biases[user_id])
                self.item_biases[item_id] += self.lr * (error - self.reg * self.item_biases[item_id])
                
                # Update user and item factors
                # We need to save a copy of user factors before updating for computing item factors
                user_factors_copy = self.user_factors[user_idx].copy()
                
                # Update user factors
                # For each factor k:
                # new_user_factor_k = user_factor_k + lr * (error * item_factor_k - reg * user_factor_k)
                # For example with factor 0:
                # new_user_factor_0 = 0.5 + 0.005 * (0.5 * 0.8 - 0.02 * 0.5) = 0.5 + 0.005 * (0.4 - 0.01) = 0.501950
                self.user_factors[user_idx] += self.lr * (error * self.item_factors[item_idx] - 
                                                         self.reg * self.user_factors[user_idx])
                
                # Update item factors
                # Similar update as user factors, but using the user_factors_copy to ensure consistent update
                self.item_factors[item_idx] += self.lr * (error * user_factors_copy -
                                                         self.reg * self.item_factors[item_idx])
            
            # Print epoch progress
            rmse = math.sqrt(squared_error / len(shuffled_data))  # -> Calculate Root Mean Square Error
            print(f"Epoch {epoch+1}/{self.n_epochs}: RMSE = {rmse:.4f}")
            
            # Adaptive learning rate - reduce over time
            self.lr *= 0.9                                # -> Decrease learning rate by 10% each epoch
        
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

        """
        Example:
            For User 101 and Movie 203:
            
            self.global_mean = 3.67
            self.user_biases[101] = 0.33
            self.item_biases[203] = 0.0
            self.user_factors[user_idx] = [0.5, 0.8]
            self.item_factors[item_idx] = [0.8, 0.1]
            
            prediction = 3.67 + 0.33 + 0.0 + dot([0.5, 0.8], [0.8, 0.1])
                      = 3.67 + 0.33 + 0.0 + (0.5*0.8 + 0.8*0.1)
                      = 4.0 + (0.4 + 0.08)
                      = 4.48
                      
            Final prediction after clipping to [1, 5] range: 4.48
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
    
    """
    | Agorithms                       | RMSE   | MAE       |
    |---------------------------------|--------|-----------|
    | Random                          | 1.4385 |    1.1478 |
    | Collaborative Filtering         |--------|-----------|
    | SVD                             | 0.9187 |    0.7112 |   
    """

    # Run evaluation
    print("Evaluating algorithms...")
    evaluator.Evaluate(False)
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)