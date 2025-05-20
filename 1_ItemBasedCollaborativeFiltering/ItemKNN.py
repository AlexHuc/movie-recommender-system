# -*- coding: utf-8 -*-
"""
Custom Item-Based Collaborative Filtering KNN Algorithm Implementation 
without Surprise but compatible with Framework evaluation
"""
import sys
import os
# Get the absolute path to the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Libs used for PureItemKNN
from EvaluationFramework.MovieLens import MovieLens
import numpy as np
import math
import heapq

# Libs used for AdaptedItemKNN
from surprise import AlgoBase, PredictionImpossible

# Libs used for Evaluation
from utils.LoadMovieLensData import LoadMovieLensData
from EvaluationFramework.Evaluator import Evaluator
from surprise import NormalPredictor
import random


class PureItemKNN:
    """
    A pure Python implementation of item-based collaborative filtering KNN algorithm
    """
    
    def __init__(self, k=40, sim_options={'name': 'pearson', 'user_based': False}):
        """
        Initialize with k nearest neighbors and similarity options
        
        Args:
            k (int): Number of neighbors to use in prediction
            sim_options (dict): Options for similarity calculation
                - name: 'pearson' or 'cosine'
                - user_based: False (item-based approach)
        """
        self.k = k
        self.similarity_name = sim_options.get('name', 'pearson')
        self.similarities = None   # Will store item-item similarity matrix
        self.item_means = {}       # Mean rating for each item
        self.item_index_to_id = {}
        self.item_id_to_index = {}
        self.item_ratings = {}     # Dictionary of item_id -> {user_id -> rating}
        self.global_mean = 0       # Global mean of all ratings
    
    def fit(self, ratings_data):
        """
        Train the algorithm using pure Python (no external ML libraries)
        
        Args:
            ratings_data: List of (user_id, item_id, rating) tuples
        """
        print("Processing ratings data for item-based collaborative filtering...")
        
        # Extract all items and build mapping
        all_item_ids = set()
        all_user_ids = set()
        all_ratings = []
        
        # Group ratings by item
        for user_id, item_id, rating in ratings_data:
            # Add to sets for tracking
            all_item_ids.add(item_id)
            all_user_ids.add(user_id)
            all_ratings.append(rating)
            
            # Initialize item dictionary if needed
            if item_id not in self.item_ratings:
                self.item_ratings[item_id] = {}
            
            # Store rating
            self.item_ratings[item_id][user_id] = rating
            
        # Create item ID mappings for matrix
        self.item_ids = sorted(list(all_item_ids))
        self.item_id_to_index = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        self.item_index_to_id = {idx: item_id for item_id, idx in self.item_id_to_index.items()}
        
        # Calculate global and item means
        self.global_mean = sum(all_ratings) / len(all_ratings) if all_ratings else 0
        
        for item_id, user_ratings in self.item_ratings.items():
            if user_ratings:
                self.item_means[item_id] = sum(user_ratings.values()) / len(user_ratings)
            else:
                self.item_means[item_id] = self.global_mean
        
        # Compute similarity matrix
        n_items = len(self.item_ids)
        self.similarities = np.zeros((n_items, n_items))
        
        print("Computing item-item similarity matrix...")
        # Track progress
        processed = 0
        total = (n_items * (n_items - 1)) // 2
        
        for i in range(n_items):
            if i % 100 == 0:
                progress = (processed / total) * 100
                print(f"Progress: {progress:.1f}% ({processed}/{total}) ({i} items out of {n_items})")
                
            item_id_i = self.item_index_to_id[i]
            
            for j in range(i+1, n_items):
                item_id_j = self.item_index_to_id[j]
                
                # Compute similarity
                sim = self._compute_similarity(item_id_i, item_id_j)
                
                # Store symmetrically
                self.similarities[i, j] = sim
                self.similarities[j, i] = sim
                
                processed += 1
        
        print("Item-item similarity computation complete!")
        return self
    
    def _compute_similarity(self, item1, item2):
        """
        Compute similarity between two items based on user ratings
        
        Args:
            item1: First item ID
            item2: Second item ID
            
        Returns:
            float: Similarity value
        """
        # Get ratings for each item
        ratings1 = self.item_ratings.get(item1, {})
        ratings2 = self.item_ratings.get(item2, {})
        
        # Find common users who rated both items
        common_users = set(ratings1.keys()) & set(ratings2.keys())
        
        # Need at least 2 common users to compute meaningful similarity
        if len(common_users) < 2:
            return 0
        
        if self.similarity_name == 'cosine':
            return self._compute_cosine_similarity(ratings1, ratings2, common_users)
        else:  # Default to Pearson
            return self._compute_pearson_similarity(ratings1, ratings2, common_users, item1, item2)
    
    def _compute_cosine_similarity(self, ratings1, ratings2, common_users):
        """Compute cosine similarity between two sets of ratings"""
        # Extract ratings for common users
        vec1 = [ratings1[u] for u in common_users]
        vec2 = [ratings2[u] for u in common_users]
        
        # Compute dot product
        dot_product = sum(r1 * r2 for r1, r2 in zip(vec1, vec2))
        
        # Compute magnitudes
        mag1 = math.sqrt(sum(r * r for r in vec1))
        mag2 = math.sqrt(sum(r * r for r in vec2))
        
        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0
        
        return dot_product / (mag1 * mag2)
    
    def _compute_pearson_similarity(self, ratings1, ratings2, common_users, item1, item2):
        """Compute Pearson correlation between two sets of ratings"""
        # Get mean ratings
        mean1 = self.item_means.get(item1, self.global_mean)
        mean2 = self.item_means.get(item2, self.global_mean)
        
        # Calculate numerator and denominators
        numerator = 0
        denom1 = 0
        denom2 = 0
        
        for user in common_users:
            # Calculate deviation from mean
            dev1 = ratings1[user] - mean1
            dev2 = ratings2[user] - mean2
            
            numerator += dev1 * dev2
            denom1 += dev1 * dev1
            denom2 += dev2 * dev2
        
        # Avoid division by zero
        if denom1 == 0 or denom2 == 0:
            return 0
        
        return numerator / (math.sqrt(denom1) * math.sqrt(denom2))
    
    def predict(self, user_id, item_id):
        """
        Predict rating for a user-item pair using item-based CF
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating or None if prediction impossible
        """
        # Check if item is known
        if item_id not in self.item_id_to_index:
            return None
        
        # Get index for target item
        item_idx = self.item_id_to_index[item_id]
        
        # Find items rated by this user
        rated_items = []
        for i_id, user_ratings in self.item_ratings.items():
            if user_id in user_ratings and i_id in self.item_id_to_index:
                i_idx = self.item_id_to_index[i_id]
                similarity = self.similarities[item_idx, i_idx]
                if similarity > 0:  # Only use positively correlated items
                    rating = user_ratings[user_id]
                    rated_items.append((similarity, i_id, rating))
        
        if not rated_items:
            # If no similar items found, return the item's mean rating
            return self.item_means.get(item_id, self.global_mean)
        
        # Get top-k neighbors
        k_neighbors = heapq.nlargest(min(self.k, len(rated_items)), rated_items, key=lambda x: x[0])
        
        # Calculate weighted average
        sim_total = sum(sim for sim, _, _ in k_neighbors)
        if sim_total == 0:
            return self.item_means.get(item_id, self.global_mean)
        
        # For item-based CF, we can use either weighted average or adjusted weighted average
        # Here we use adjusted weighted average accounting for item mean differences
        target_mean = self.item_means.get(item_id, self.global_mean)
        weighted_sum = 0
        
        for sim, i_id, rating in k_neighbors:
            # Adjust for item mean difference
            i_mean = self.item_means.get(i_id, self.global_mean)
            weighted_sum += sim * (rating - i_mean)
        
        # Predict as target item's mean plus weighted deviations
        predicted_rating = target_mean + (weighted_sum / sim_total)
        
        # Clip to valid rating range (typically 1-5)
        predicted_rating = max(1.0, min(5.0, predicted_rating))
        
        return predicted_rating


class AdaptedItemKNN(AlgoBase):
    """
    Adapter class that wraps PureItemKNN to make it compatible with Surprise's AlgoBase
    for evaluation with the Framework
    """
    
    def __init__(self, k=40, sim_options=None):
        """Initialize with k nearest neighbors and similarity options"""
        AlgoBase.__init__(self)
        if sim_options is None:
            sim_options = {'name': 'pearson', 'user_based': False}
        self.k = k
        self.sim_options = sim_options
        self.pure_item_knn = PureItemKNN(k=k, sim_options=sim_options)
        
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
        
        # Train our pure Python item-based KNN implementation
        self.pure_item_knn.fit(ratings_data)
        
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
            
            # Get prediction from pure item-based KNN
            prediction = self.pure_item_knn.predict(user_id, item_id)
            
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
    itemKNN_pearson = AdaptedItemKNN(k=40, sim_options={'name': 'pearson', 'user_based': False})
    evaluator.AddAlgorithm(itemKNN_pearson, "ItemKNN-Pearson")
    
    itemKNN_cosine = AdaptedItemKNN(k=40, sim_options={'name': 'cosine', 'user_based': False})
    evaluator.AddAlgorithm(itemKNN_cosine, "ItemKNN-Cosine")
    
    random_predictor = NormalPredictor()
    evaluator.AddAlgorithm(random_predictor, "Random")
    
    # Run evaluation
    print("Evaluating algorithms...")
    evaluator.Evaluate(False)
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)