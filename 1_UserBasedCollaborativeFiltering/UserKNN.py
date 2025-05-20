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
from EvaluationFramework.MovieLens import MovieLens
import numpy as np
import math
import heapq

# Libs used for AdaptedUserKNN
from surprise import AlgoBase, PredictionImpossible

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
        self.similarities = None  # Will store user-user similarity matrix
        self.user_means = {}  # Mean rating for each user
        self.user_index_to_id = {}
        self.user_id_to_index = {}
        self.user_ratings = {}  # Dictionary of user_id -> {item_id -> rating}
        self.items_rated_by_user = {}  # Dictionary of item_id -> set of user_ids
        self.global_mean = 0  # Global mean of all ratings
    
    def fit(self, ratings_data):
        """
        Train the algorithm using pure Python (no external ML libraries)
        
        Args:
            ratings_data: List of (user_id, item_id, rating) tuples
        """
        print("Processing ratings data for user-based collaborative filtering...")
        
        # Extract all users and build mapping
        all_user_ids = set()
        all_item_ids = set()
        all_ratings = []
        
        # Group ratings by user
        for user_id, item_id, rating in ratings_data:
            # Add to sets for tracking
            all_user_ids.add(user_id)
            all_item_ids.add(item_id)
            all_ratings.append(rating)
            
            # Initialize user dictionary if needed
            if user_id not in self.user_ratings:
                self.user_ratings[user_id] = {}
            
            # Store rating
            self.user_ratings[user_id][item_id] = rating
            
            # Track which users rated each item (for faster lookups)
            if item_id not in self.items_rated_by_user:
                self.items_rated_by_user[item_id] = set()
            self.items_rated_by_user[item_id].add(user_id)
            
        # Create user ID mappings for matrix
        self.user_ids = sorted(list(all_user_ids))
        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.user_index_to_id = {idx: user_id for user_id, idx in self.user_id_to_index.items()}
        
        # Calculate global and user means
        self.global_mean = sum(all_ratings) / len(all_ratings) if all_ratings else 0
        
        for user_id, item_ratings in self.user_ratings.items():
            if item_ratings:
                self.user_means[user_id] = sum(item_ratings.values()) / len(item_ratings)
            else:
                self.user_means[user_id] = self.global_mean
        
        # Compute similarity matrix
        n_users = len(self.user_ids)
        self.similarities = np.zeros((n_users, n_users))
        
        print("Computing user-user similarity matrix...")
        # Track progress
        processed = 0
        total = (n_users * (n_users - 1)) // 2
        
        for i in range(n_users):
            if i % 100 == 0:
                progress = (processed / total) * 100
                print(f"Progress: {progress:.1f}% ({processed}/{total}) ({i} users out of {n_users})")
                
            user_id_i = self.user_index_to_id[i]
            
            for j in range(i+1, n_users):
                user_id_j = self.user_index_to_id[j]
                
                # Compute similarity
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
        ratings1 = self.user_ratings.get(user1, {})
        ratings2 = self.user_ratings.get(user2, {})
        
        # Find common items rated by both users
        common_items = set(ratings1.keys()) & set(ratings2.keys())
        
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
        vec1 = [ratings1[i] for i in common_items]
        vec2 = [ratings2[i] for i in common_items]
        
        # Compute dot product
        dot_product = sum(r1 * r2 for r1, r2 in zip(vec1, vec2))
        
        # Compute magnitudes
        mag1 = math.sqrt(sum(r * r for r in vec1))
        mag2 = math.sqrt(sum(r * r for r in vec2))
        
        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0
        
        return dot_product / (mag1 * mag2)
    
    def _compute_pearson_similarity(self, ratings1, ratings2, common_items, user1, user2):
        """Compute Pearson correlation between two sets of ratings"""
        # Get mean ratings
        mean1 = self.user_means.get(user1, self.global_mean)
        mean2 = self.user_means.get(user2, self.global_mean)
        
        # Calculate numerator and denominators
        numerator = 0
        denom1 = 0
        denom2 = 0
        
        for item in common_items:
            # Calculate deviation from mean
            dev1 = ratings1[item] - mean1
            dev2 = ratings2[item] - mean2
            
            numerator += dev1 * dev2
            denom1 += dev1 * dev1
            denom2 += dev2 * dev2
        
        # Avoid division by zero
        if denom1 == 0 or denom2 == 0:
            return 0
        
        return numerator / (math.sqrt(denom1) * math.sqrt(denom2))
    
    def predict(self, user_id, item_id):
        """
        Predict rating for a user-item pair using user-based CF
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating or None if prediction impossible
        """
        # Check if user is known
        if user_id not in self.user_id_to_index:
            # For unknown users, return global mean or item average if available
            return self.global_mean
        
        # Get index for target user
        user_idx = self.user_id_to_index[user_id]
        
        # Check if user has already rated this item
        if user_id in self.user_ratings and item_id in self.user_ratings[user_id]:
            # Return the actual rating (useful for evaluation)
            return self.user_ratings[user_id][item_id]
        
        # Get users who rated this item
        neighbors = []
        
        # Using the inverted index for faster lookup
        if item_id in self.items_rated_by_user:
            for other_user_id in self.items_rated_by_user[item_id]:
                # Skip if it's the target user
                if other_user_id == user_id:
                    continue
                    
                # Get ratings and similarity
                if other_user_id in self.user_id_to_index:
                    other_idx = self.user_id_to_index[other_user_id]
                    similarity = self.similarities[user_idx, other_idx]
                    
                    if similarity > 0:  # Only consider positively correlated users
                        rating = self.user_ratings[other_user_id][item_id]
                        neighbors.append((similarity, other_user_id, rating))
        
        if not neighbors:
            # If no neighbors found, return the user's mean rating
            return self.user_means.get(user_id, self.global_mean)
        
        # Get top-k neighbors
        k_neighbors = heapq.nlargest(min(self.k, len(neighbors)), neighbors, key=lambda x: x[0])
        
        # Calculate weighted average
        sim_total = sum(sim for sim, _, _ in k_neighbors)
        if sim_total == 0:
            return self.user_means.get(user_id, self.global_mean)
        
        # For user-based CF, we use normalized weighted average
        user_mean = self.user_means.get(user_id, self.global_mean)
        weighted_sum = 0
        
        for sim, neighbor_id, rating in k_neighbors:
            # Adjust for user mean difference
            neighbor_mean = self.user_means.get(neighbor_id, self.global_mean)
            weighted_sum += sim * (rating - neighbor_mean)
        
        # Predict as user's mean plus weighted deviations
        predicted_rating = user_mean + (weighted_sum / sim_total)
        
        # Clip to valid rating range (typically 1-5)
        predicted_rating = max(1.0, min(5.0, predicted_rating))
        
        return predicted_rating


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
    
    # Run evaluation
    print("Evaluating algorithms...")
    evaluator.Evaluate(False)
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)