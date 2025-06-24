# -*- coding: utf-8 -*-
"""
Custom User-Based Collaborative Filtering KNN Algorithm Implementation 
without Surprise but compatible with Framework evaluation
"""
import sys
import os
# Get the absolute path to the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Libs used for PureUserKNN
import numpy as np
import math
import heapq

# Libs used for AdaptedUserKNN
from surprise import AlgoBase, PredictionImpossible

# Libs to save and load models
from datetime import datetime
import pickle

# Libs used for Evaluation
from utils.LoadMovieLensData import LoadMovieLensData
from EvaluationFramework.Evaluator import Evaluator
from surprise import NormalPredictor
import random


class PureUserKNN:
    """
    A pure Python implementation of user-based collaborative filtering KNN algorithm
    """
    
    def __init__(self, k=40, sim_options={'name': 'pearson', 'user_based': True}):
        """
        Initialize with k nearest neighbors and similarity options
        
        Args:
            k (int): Number of neighbors to use in prediction
            sim_options (dict): Options for similarity calculation
                - name: 'pearson' or 'cosine'
                - user_based: True (user-based approach)
        """
        self.k = k
        self.similarity_name = sim_options.get('name', 'pearson')
        self.similarities = None                           # SIMILARITY MATRIX: Will store user-user similarity matrix
        self.user_means = {}                               # Mean rating for each user
        self.user_index_to_id = {}                         # Dictionary of index -> user_id
        self.user_id_to_index = {}                         # Dictionary of user_id -> index
        self.user_ratings = {}                             # Dictionary of user_id -> {item_id -> rating}
        self.items_rated_by_user = {}                      # Dictionary of item_id -> set of user_ids
        self.global_mean = 0                               # Global mean of all ratings (default rating)
    

    def fit(self, ratings_data):
        """
        Train the algorithm using pure Python (no external ML libraries)
        
        Args:
            ratings_data: List of (user_id, item_id, rating) tuples
        """
        
        """
        Example:
            User 101: Movie 201 (5), Movie 202 (3), Movie 204 (4)
            User 102: Movie 201 (4), Movie 203 (5), Movie 205 (3)
            User 103: Movie 202 (4), Movie 203 (2), Movie 204 (3)
            User 104: Movie 201 (3), Movie 202 (2), Movie 205 (5)
            User 105: Movie 203 (4), Movie 204 (5), Movie 205 (4)
            User 106: Movie 201 (5), Movie 203 (3), Movie 204 (4)

        self.user_ratings = {
            101: {201: 5, 202: 3, 204: 4},        # User 101 ratings
            102: {201: 4, 203: 5, 205: 3},        # User 102 ratings 
            103: {202: 4, 203: 2, 204: 3},        # User 103 ratings
            104: {201: 3, 202: 2, 205: 5},        # User 104 ratings
            105: {203: 4, 204: 5, 205: 4},        # User 105 ratings
            106: {201: 5, 203: 3, 204: 4}         # User 106 ratings
        }

        self.user_ids = [101, 102, 103, 104, 105, 106]

        self.user_id_to_index = {101: 0, 102: 1, 103: 2, 104: 3, 105: 4, 106: 5}

        self.user_index_to_id = {0: 101, 1: 102, 2: 103, 3: 104, 4: 105, 5: 106}
        
        all_ratings = [5, 3, 4, 4, 5, 3, 4, 2, 3, 3, 2, 5, 4, 5, 4, 5, 3, 4]
        self.global_mean = 3.83                          # Average of all ratings

        self.user_means = {
            101: 4.00,  # (5+3+4)/3
            102: 4.00,  # (4+5+3)/3
            103: 3.00,  # (4+2+3)/3
            104: 3.33,  # (3+2+5)/3
            105: 4.33,  # (4+5+4)/3
            106: 4.00   # (5+3+4)/3
        }
        
        self.items_rated_by_user = {
            201: {101, 102, 104, 106},  # Users who rated Movie 201
            202: {101, 103, 104},       # Users who rated Movie 202
            203: {102, 103, 105, 106},  # Users who rated Movie 203
            204: {101, 103, 105, 106},  # Users who rated Movie 204
            205: {102, 104, 105}        # Users who rated Movie 205
        }

        # Initializing similarity matrix with zeros
        self.similarities = np.zeros((len(self.user_ids), len(self.user_ids))) # -> Create a square matrix of zeros with size n x n, where n is the number of users

        # Computed pearson similarities between users
        self.similarities = [
            [ 1.000  0.500  0.866  0.500  0.866  1.000]
            [ 0.500  1.000  0.000 -0.500  0.500  0.500]
            [ 0.866  0.000  1.000  0.000  0.866  0.866]
            [ 0.500 -0.500  0.000  1.000  0.500  0.500]
            [ 0.866  0.500  0.866  0.500  1.000  0.866]
            [ 1.000  0.500  0.866  0.500  0.866  1.000]
        ]

        # Compute cosine similarities between users
        self.similarities = [
            [ 1.000  0.577  0.707  0.95   0.707  1.000]
            [ 0.577  1.000  0.000 -0.577  0.577  0.577]
            [ 0.707  0.000  1.000  0.000  0.707  0.707]
            [ 0.95  -0.577  0.000  1.000  0.577  0.577]
            [ 0.707  0.577  0.707  0.577  1.000  0.707]
            [ 1.000  0.577  0.707  0.577  0.707  0.100]
        ]
        """
        print("Processing ratings data for user-based collaborative filtering...")
        
        # Extract all users and build mapping
        all_user_ids = set()                                       # -> Create an empty set to track unique user IDs
        all_item_ids = set()                                       # -> Create an empty set to track unique item IDs
        all_ratings = []                                           # -> Create an empty list to store all ratings
        
        # Group ratings by user
        for user_id, item_id, rating in ratings_data:              # -> Iterate through each (user_id, item_id, rating) tuple
            # Add to sets for tracking
            all_user_ids.add(user_id)                              # -> Add user_id to all_user_ids set
            all_item_ids.add(item_id)                              # -> Add item_id to all_item_ids set
            all_ratings.append(rating)                             # -> Append rating to all_ratings list
            
            # Initialize user dictionary if needed
            if user_id not in self.user_ratings:
                self.user_ratings[user_id] = {}                    # -> Initialize an empty dictionary if user_id not in user_ratings
            
            # Store rating
            self.user_ratings[user_id][item_id] = rating           # -> Store the rating that the user gave to the item
            
            # Track which users rated each item (for faster lookups)
            if item_id not in self.items_rated_by_user:
                self.items_rated_by_user[item_id] = set()          # -> Initialize an empty set if item_id not in items_rated_by_user
            self.items_rated_by_user[item_id].add(user_id)         # -> Add user_id to the set of users who rated the item
            
        # Create user ID mappings for matrix
        self.user_ids = sorted(list(all_user_ids))                                               # -> Convert the all_user_ids set to a sorted list
        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}      # -> Create a mapping from user_id to index
        self.user_index_to_id = {idx: user_id for user_id, idx in self.user_id_to_index.items()} # -> Create a mapping from index to user_id
        
        # Calculate global and user means
        self.global_mean = sum(all_ratings) / len(all_ratings) if all_ratings else 0      # -> Calculate the global mean rating from all ratings
        
        for user_id, item_ratings in self.user_ratings.items():                           # -> Iterate through each user_id and their item ratings
            if item_ratings:                                                              # -> Check if there are item ratings for the user
                self.user_means[user_id] = sum(item_ratings.values()) / len(item_ratings) # -> Calculate the mean rating for the user
            else:                                                                         # -> If no ratings then:
                self.user_means[user_id] = self.global_mean                               # -> Set user mean to global mean
        
        # Compute similarity matrix
        n_users = len(self.user_ids)                               # -> Get the number of unique users
        self.similarities = np.zeros((n_users, n_users))           # -> Initialize a square matrix n x n with zeros to store user-user similarities
        
        # ----------------------------------------------------------------
        # Just to track progress in the console
        print("Computing user-user similarity matrix...")
        processed = 0
        total = (n_users * (n_users - 1)) // 2                     # -> Formula for the total number of user pairs to process
        # ----------------------------------------------------------------
        
        for i in range(n_users):                                   # -> Iterate through each user index
            # ----------------------------------------------------------------
            # Just to track progress in the console
            if i % 100 == 0:
                progress = (processed / total) * 100
                print(f"Progress: {progress:.1f}% ({processed}/{total}) ({i} users out of {n_users})")
            # ----------------------------------------------------------------
                
            user_id_i = self.user_index_to_id[i]                   # -> Get the user ID for the current index
            
            for j in range(i+1, n_users):                          # -> Iterate through the remaining users to compute similarity
                user_id_j = self.user_index_to_id[j]               # -> Get the user ID for the current index
                
                # Compute similarity between user_id_i and user_id_j
                sim = self._compute_similarity(user_id_i, user_id_j)
                
                # Store symmetrically
                self.similarities[i, j] = sim
                self.similarities[j, i] = sim
                
                processed += 1
        
        print("User-user similarity computation complete!")
        return self
    

    def _compute_similarity(self, user1, user2):
        """
        Compute similarity between two users based on their ratings
        
        Args:
            user1: First user ID
            user2: Second user ID
            
        Returns:
            float: Similarity value
        """
        # Get ratings for each user
        ratings1 = self.user_ratings.get(user1, {})                # -> Get the ratings for user 101, default to empty dict if not found, ex: ratings1 = {201: 5, 202: 3, 204: 4}
        ratings2 = self.user_ratings.get(user2, {})                # -> Get the ratings for user 104, default to empty dict if not found, ex: ratings2 = {201: 3, 202: 2, 205: 5}
        
        # Find common items rated by both users
        common_items = set(ratings1.keys()) & set(ratings2.keys()) # -> Find the intersection of item IDs that both users rated, ex: common_items = {201, 202}
        
        # Need at least 2 common items to compute meaningful similarity
        if len(common_items) < 2:
            return 0
        
        if self.similarity_name == 'cosine':
            return self._compute_cosine_similarity(ratings1, ratings2, common_items)
        else:  # Default to Pearson
            return self._compute_pearson_similarity(ratings1, ratings2, common_items, user1, user2)
    

    def _compute_cosine_similarity(self, ratings1, ratings2, common_items):
        """Compute cosine similarity between two sets of ratings"""
        # Extract ratings for common items
        vec1 = [ratings1[i] for i in common_items]                 # -> Create a vector of ratings for user1 from common items, ex: vec1 = [5, 3, 4]
        vec2 = [ratings2[i] for i in common_items]                 # -> Create a vector of ratings for user2 from common items, ex: vec2 = [3, 2, 5]
        
        # Compute dot product
        dot_product = sum(r1 * r2 for r1, r2 in zip(vec1, vec2))   # -> Calculate the dot product of the two vectors, ex: dot_product = 5*3 + 3*2 + 4*5 = 15 + 6 + 20 = 41
        # Compute magnitudes
        mag1 = math.sqrt(sum(r * r for r in vec1))                 # -> Calculate the magnitude of vec1, ex: mag1 = sqrt(5^2 + 3^2 + 4^2) = sqrt(50) = 7.07
        mag2 = math.sqrt(sum(r * r for r in vec2))                 # -> Calculate the magnitude of vec2, ex: mag2 = sqrt(3^2 + 2^2 + 5^2) = sqrt(38) = 6.16
        
        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0
        
        return dot_product / (mag1 * mag2)                         # -> Calculate cosine similarity as dot_product / (mag1 * mag2), ex: cosine_similarity = 41 / (7.07 * 6.16) = 0.95
    

    def _compute_pearson_similarity(self, ratings1, ratings2, common_items, user1, user2):
        """Compute Pearson correlation between two sets of ratings"""

        """
            # ratings1 = {201: 5, 202: 3, 204: 4}                  # User 101
            # ratings2 = {201: 3, 202: 2, 205: 5}                  # User 104

            # common_items = {201, 202}                            # Common items rated by both users

            # mean1 = 4.00                                         # Mean rating for user 101
            # mean2 = 3.33                                         # Mean rating for user 104

            # First iteration:
                dev1 = 5 - 4.00 = 1.00                             # Deviation for item 201 for user 101
                dev2 = 3 - 3.33 = -0.33                            # Deviation for item 201 for user 104
                numerator = 0.00 + (1.00 * -0.33) = -0.33          # Update numerator
                denom1 = 0.00 + (1.00 * 1.00) = 1.00               # Update denominator for user 101
                denom2 = 0.00 + (-0.33 * -0.33) = 0.11             # Update denominator for user 104
            # Second iteration:
                dev1 = 3 - 4.00 = -1.00                            # Deviation for item 202 for user 101
                dev2 = 2 - 3.33 = -1.33                            # Deviation for item 202 for user 104
                numerator = -0.33 + (-1.00 * -1.33) = 1.00         # Update numerator
                denom1 = 1.00 + (-1.00 * -1.00) = 2.00             # Update denominator for user 101
                denom2 = 0.11 + (-1.33 * -1.33) = 1.78             # Update denominator for user 104
            # Final calculation:
                numerator = 1.00                                   # Final numerator
                denom1 = 2.00                                      # Final denominator for user 101
                denom2 = 1.78                                      # Final denominator for user 104
                
            similarity = 1.00 / (sqrt(2.00) * sqrt(1.78)) = 0.500
        """

        # Get mean ratings
        mean1 = self.user_means.get(user1, self.global_mean)       # -> Get the mean rating for user1, default to global mean if not found, ex: mean1 = 4.0
        mean2 = self.user_means.get(user2, self.global_mean)       # -> Get the mean rating for user2, default to global mean if not found, ex: mean2 = 3.33
        
        # Calculate numerator and denominators
        numerator = 0                                              # -> Initialize numerator for Pearson correlation
        denom1 = 0                                                 # -> Initialize denominator for user1
        denom2 = 0                                                 # -> Initialize denominator for user2
        
        for item in common_items:
            # Calculate deviation from mean
            dev1 = ratings1[item] - mean1                          # -> Calculate deviation of user1's rating for item from their mean, ex: dev1 = 5 - 4.0 = 1.0
            dev2 = ratings2[item] - mean2                          # -> Calculate deviation of user2's rating for item from their mean, ex: dev2 = 3 - 3.33 = -0.33
            
            numerator += dev1 * dev2                               # -> Update numerator with product of deviations, ex: numerator = 0.0 + (1.0 * -0.33) = -0.33
            denom1 += dev1 * dev1                                  # -> Update denominator for user1 with square of deviation, ex: denom1 = 0.0 + (1.0 * 1.0) = 1.0
            denom2 += dev2 * dev2                                  # -> Update denominator for user2 with square of deviation, ex: denom2 = 0.0 + (-0.33 * -0.33) = 0.11
        
        # Avoid division by zero
        if denom1 == 0 or denom2 == 0:
            return 0
        
        return numerator / (math.sqrt(denom1) * math.sqrt(denom2)) # -> Calculate Pearson correlation as numerator / (sqrt(denom1) * sqrt(denom2)), ex: pearson_similarity = 1.0 / (sqrt(2.0) * sqrt(1.78)) = 0.500
    

    def predict(self, user_id, item_id):
        """
        Predict rating for a user-item pair using user-based CF
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating or None if prediction impossible
        """

        """
            We computed the following user-user similarity matrix:
            self.similarities = [
                [ 1.000  0.500  0.866  0.500  0.866  1.000]
                [ 0.500  1.000  0.000 -0.500  0.500  0.500]
                [ 0.866  0.000  1.000  0.000  0.866  0.866]
                [ 0.500 -0.500  0.000  1.000  0.500  0.500]
                [ 0.866  0.500  0.866  0.500  1.000  0.866]
                [ 1.000  0.500  0.866  0.500  0.866  1.000]
            ]

            Example:
            User 101: Movie 201 (5), Movie 202 (3), Movie 204 (4)
            User 102: Movie 201 (4), Movie 203 (5), Movie 205 (3)
            User 103: Movie 202 (4), Movie 203 (2), Movie 204 (3)
            User 104: Movie 201 (3), Movie 202 (2), Movie 205 (5)
            User 105: Movie 203 (4), Movie 204 (5), Movie 205 (4)
            User 106: Movie 201 (5), Movie 203 (3), Movie 204 (4)

            We want to predict the rating for User 101 on Movie 203

            1. Check if User 101 is known (exists in user_id_to_index)
            2. Get index for User 101 (let's say it's index 0)
            3. Find users who rated Movie 203:
                - User 102 -> similarity = 0.500
                - User 103 -> similarity = 0.866
                - User 105 -> similarity = 0.866
                - User 106 -> similarity = 1.000
            4. Get top-k neighbors (let's say k=2):
                - User 106 (similarity = 1.000, rating = 3)
                - User 103 (similarity = 0.866, rating = 2)
            5. Calculate weighted average:
                weighted_sum = (1.000 * (3 - 4.00)) + (0.866 * (2 - 3.00)) = -1.0 - 0.866 = -1.866
                sim_total = 1.000 + 0.866 = 1.866
                predicted_rating = 4.00 + (-1.866 / 1.866) = 4.00 - 1.00 = 3.00
        """

        # Check if user is known
        if user_id not in self.user_id_to_index:                   # -> Check if the user_id is in the user_id_to_index mapping
            # For unknown users, return global mean or item average if available
            return self.global_mean
        
        # Get index for target user
        user_idx = self.user_id_to_index[user_id]                  # -> Get the index of the user_id in the user_id_to_index mapping, ex: user_idx = 0 for User 101
        
        # Check if user has already rated this item
        if user_id in self.user_ratings and item_id in self.user_ratings[user_id]:
            # Return the actual rating (useful for evaluation)
            return self.user_ratings[user_id][item_id]             # -> If user has already rated this item, return that rating
        
        # Get users who rated this item
        neighbors = []
        
        # Using the inverted index for faster lookup
        if item_id in self.items_rated_by_user:                               # -> Check if the item has been rated by any user
            for other_user_id in self.items_rated_by_user[item_id]:           # -> Iterate through each user who rated the item
                # Skip if it's the target user
                if other_user_id == user_id:
                    continue
                    
                # Get ratings and similarity
                if other_user_id in self.user_id_to_index:                    # -> Check if the other user is known
                    other_idx = self.user_id_to_index[other_user_id]
                    similarity = self.similarities[user_idx, other_idx]       # -> Get the similarity between target user and other user, ex: similarity = 0.500 for User 101 and User 102
                    
                    if similarity > 0:                                        # -> Only consider positively correlated users
                        rating = self.user_ratings[other_user_id][item_id]    # -> Get the rating the other user gave to the item, ex: rating = 5 for User 102 on Movie 203
                        neighbors.append((similarity, other_user_id, rating)) # -> Append a tuple of (similarity, user_id, rating) for users who rated the item, ex: neighbors = [(0.5, 102, 5), (0.866, 103, 2), (0.866, 105, 4), (1.000, 106, 3)]
        
        if not neighbors:
            # If no neighbors found, return the user's mean rating
            return self.user_means.get(user_id, self.global_mean)  # -> If no neighbors found, return the user's mean rating or global mean if not available, ex: return 4.00 for User 101
        
        # Get top-k neighbors
        # This function returns the k largest items from the neighbors list based on the similarity value (first element of the tuple)
        # If we have fewer than k neighbors, we just use what we have
        # For this example let's say self.k = 2
        # min(self.k, len(neighbors)) -> This ensures we don't try to get more neighbors than we have, ex: if k=40 and we have only 4 neighbors, it will take min(2, 4) = 4
        # key=lambda x: x[0]          -> This specifies that we want to sort by the first element of the tuple (similarity)
        k_neighbors = heapq.nlargest(min(self.k, len(neighbors)), neighbors, key=lambda x: x[0]) # -> Get the top-k neighbors based on similarity, ex: k_neighbors = [(1.000, 106, 3), (0.866, 103, 2), (0.866, 105, 4), (0.500, 102, 5)]
        
        # Calculate weighted average
        sim_total = sum(sim for sim, _, _ in k_neighbors)          # -> Calculate the total similarity of the top-k neighbors, ex: sim_total = 1.000 + 0.866 + 0.866 + 0.500 = 3.232
        if sim_total == 0:
            return self.user_means.get(user_id, self.global_mean)  # -> If total similarity is zero, return the user's mean rating or global mean if not available, ex: return 4.00 for User 101
        
        # For user-based CF, we use normalized weighted average
        user_mean = self.user_means.get(user_id, self.global_mean) # -> Get the mean rating for the target user, ex: user_mean = 4.00 for User 101
        weighted_sum = 0                                           # -> Initialize weighted sum for the prediction
        
        for sim, neighbor_id, rating in k_neighbors:
            # Adjust for user mean difference 
            neighbor_mean = self.user_means.get(neighbor_id, self.global_mean) # -> Get the mean rating for the neighbor user, ex: neighbor_mean = 4.00 for User 106
            weighted_sum += sim * (rating - neighbor_mean)         # -> Calculate the weighted sum of deviations from the neighbor user mean, ex: weighted_sum = (1.000 * (3 - 4.00)) + (0.866 * (2 - 3.00)) + (0.866 * (4 - 4.33)) + (0.500 * (5 - 4.00)) = -1.0 - 0.866 - 0.286 + 0.5 = -1.652
        
        # Predict as user's mean plus weighted deviations
        predicted_rating = user_mean + (weighted_sum / sim_total)  # -> Calculate the predicted rating as user's mean plus the weighted average of deviations, ex: predicted_rating = 4.00 + (-1.652 / 3.232) = 4.00 - 0.511 = 3.489
        
        # Clip to valid rating range (typically 1-5)
        predicted_rating = max(1.0, min(5.0, predicted_rating))    # -> Ensure the predicted rating is within the valid range, ex: predicted_rating = max(1.0, min(5.0, 3.489)) = 3.489
        
        return predicted_rating                                    # -> Return the predicted rating for the user-item pair, ex: return 3.489 for User 101 on Movie 203

class AdaptedUserKNN(AlgoBase):
    """
    Adapter class that wraps PureUserKNN to make it compatible with Surprise's AlgoBase
    for evaluation with the Framework
    """
    
    def __init__(self, k=40, sim_options=None):
        """Initialize with k nearest neighbors and similarity options"""
        AlgoBase.__init__(self)
        if sim_options is None:
            sim_options = {'name': 'pearson', 'user_based': True}
        self.k = k
        self.sim_options = sim_options
        self.pure_user_knn = PureUserKNN(k=k, sim_options=sim_options)
        
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
        
        # Train our pure Python user-based KNN implementation
        self.pure_user_knn.fit(ratings_data)
        
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
            
            # Get prediction from pure user-based KNN
            prediction = self.pure_user_knn.predict(user_id, item_id)
            
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
            sim_name = self.sim_options.get('name', 'pearson')
            filename = f"../models/1_UserBasedCollaborativeFiltering/adapted_user_knn_{sim_name}_model_{timestamp}.pkl"
        
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
    userKNN_pearson = AdaptedUserKNN(k=40, sim_options={'name': 'pearson', 'user_based': True})
    evaluator.AddAlgorithm(userKNN_pearson, "UserKNN-Pearson")
    
    userKNN_cosine = AdaptedUserKNN(k=40, sim_options={'name': 'cosine', 'user_based': True})
    evaluator.AddAlgorithm(userKNN_cosine, "UserKNN-Cosine")
    
    random_predictor = NormalPredictor()
    evaluator.AddAlgorithm(random_predictor, "Random")

    """
    | Agorithms                       | RMSE   | MAE       |
    |---------------------------------|--------|-----------|
    | Random                          | 1.4385 |    1.1478 |
    | Collaborative Filtering         |--------|-----------|
    | UserKNN-Pearson                 | 0.9231 |    0.7056 |   
    | UserKNN-Cosine                  | 0.9259 |    0.7116 |  
    """
    
    # Run evaluation
    print("Evaluating algorithms...")
    evaluator.Evaluate(False)

    userKNN_pearson.save_model()
    userKNN_cosine.save_model()
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)