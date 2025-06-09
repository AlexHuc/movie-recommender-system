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
        self.similarities = None                         # SIMILARITY MATRIX: Will store item-item similarity matrix
        self.item_means = {}                             # Mean rating for each item
        self.item_index_to_id = {}                       # Dictionary of index -> item_id
        self.item_id_to_index = {}                       # Dictionary of item_id -> index
        self.item_ratings = {}                           # Dictionary of item_id -> {user_id -> rating}
        self.global_mean = 0                             # Global mean of all ratings (default rating)
    
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

        self.item_ratings = {
            201: {101: 5, 102: 4, 104: 3, 106: 5},       # Movie 201 ratings
            202: {101: 3, 103: 4, 104: 2},               # Movie 202 ratings
            203: {102: 5, 103: 2, 105: 4, 106: 3},       # Movie 203 ratings
            204: {101: 4, 103: 3, 105: 5, 106: 4},       # Movie 204 ratings
            205: {102: 3, 104: 5, 105: 4}                # Movie 205 ratings
        }

        self.item_ids = [201, 202, 203, 204, 205]

        self.item_id_to_index = {201: 0, 202: 1, 203: 2, 204: 3, 205: 4}

        self.item_index_to_id = {0: 201, 1: 202, 2: 203, 3: 204, 4: 205}
        
        all_ratings = [5, 3, 4, 4, 5, 3, 4, 2, 3, 3, 2, 5, 4, 5, 4, 5, 3, 4]
        self.global_mean = 3.83                          # Average of all ratings

        self.item_means = {
            201: 4.25,  # (5+4+3+5)/4
            202: 3.00,  # (3+4+2)/3
            203: 3.50,  # (5+2+4+3)/4
            204: 4.00,  # (4+3+5+4)/4
            205: 4.00   # (3+5+4)/3
        }

        # Initializing similarity matrix with zeros
        self.similarities = np.zeros((len(self.item_ids), len(self.item_ids))) # -> Create a square matrix of zeros with size n x n, where n is the number of items

        # Computed pearson similarities between items
        self.similarities = [
            [ 1.000  0.857 -1.000  1.000 -0.866]
            [ 0.857  1.000  0.000 -1.000  0.000]
            [-1.000  0.000  1.000  0.500  0.000]
            [ 1.000 -1.000  0.500  1.000  0.000]
            [-0.866  0.000  0.000  0.000  1.000]
        ]

        # Computed cosine similarities between items
        self.similarities = [
            [ 1.000  0.99   0.000  0.866  0.577]
            [ 0.99   1.000  0.000  0.000  0.577]
            [ 0.000  0.000  1.000  0.577  0.000]
            [ 0.866  0.000  0.577  1.000  0.000]
            [ 0.577  0.577  0.000  0.000  1.000]
        ]
        """

        print("Processing ratings data for item-based collaborative filtering...")
        
        # Extract all items and build mapping
        all_item_ids = set()                             # -> Create an empty set to track unique item IDs
        all_user_ids = set()                             # -> Create an empty set to track unique user IDs
        all_ratings = []                                 # -> Create an empty list to store all ratings
        
        # Group ratings by item
        for user_id, item_id, rating in ratings_data:    # -> Iterate through each (user_id, item_id, rating) tuple
            # Add to sets for tracking
            all_item_ids.add(item_id)                    # -> Add item_id to all_item_ids set
            all_user_ids.add(user_id)                    # -> Add user_id to all_user_ids set
            all_ratings.append(rating)                   # -> Append rating to all_ratings list
            
            # Initialize item dictionary if needed
            if item_id not in self.item_ratings:
                self.item_ratings[item_id] = {}          # -> Initialize an empty dictionary if item_id not in item_ratings
            
            # Store rating
            self.item_ratings[item_id][user_id] = rating # -> Store the rating that the user gave to the item
            
        # Create item ID mappings for matrix
        self.item_ids = sorted(list(all_item_ids))                                               # -> Convert the all_item_ids set to a sorted list
        self.item_id_to_index = {item_id: idx for idx, item_id in enumerate(self.item_ids)}      # -> Create a mapping from item_id to index
        self.item_index_to_id = {idx: item_id for item_id, idx in self.item_id_to_index.items()} # -> Create a mapping from index to item_id
        
        # Calculate global and item means
        self.global_mean = sum(all_ratings) / len(all_ratings) if all_ratings else 0 # -> Calculate the global mean rating from all ratings
        
        for item_id, user_ratings in self.item_ratings.items():                                  # -> Iterate through each item_id and its corresponding user ratings
            if user_ratings:                                                                     # -> Check if there are user ratings for the item
                self.item_means[item_id] = sum(user_ratings.values()) / len(user_ratings)        # -> Calculate the mean rating for the item
            else:                                                                                # -> If no user in the ratings then:
                self.item_means[item_id] = self.global_mean                                      # -> Set item mean to global mean
        
        # Compute similarity matrix
        n_items = len(self.item_ids)                     # -> Get the number of unique items
        self.similarities = np.zeros((n_items, n_items)) # -> Initialize a square matrix n x n with zeros to store to store item-item similarities
        
        # ----------------------------------------------------------------
        # Just to track progress in the console
        print("Computing item-item similarity matrix...")
        processed = 0
        total = (n_items * (n_items - 1)) // 2           # -> Formual for the Total number of item pairs to process
        # ----------------------------------------------------------------

        
        for i in range(n_items):                         # -> Iterate through each item index
            # ----------------------------------------------------------------
            # Just to track progress in the console
            if i % 100 == 0:
                progress = (processed / total) * 100
                print(f"Progress: {progress:.1f}% ({processed}/{total}) ({i} items out of {n_items})")
             # ----------------------------------------------------------------
                
            item_id_i = self.item_index_to_id[i]         # -> Get the item ID for the current index
            
            for j in range(i+1, n_items):                # -> Iterate through the remaining items to compute similarity
                item_id_j = self.item_index_to_id[j]     # -> Get the item ID for the current index
                
                # Compute similarity between item_id_i and item_id_j
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
        ratings1 = self.item_ratings.get(item1, {})                # -> Get the ratings for item1, default to empty dict if not found, ex: ratings1 = {101: 5, 102: 4, 104: 3, 106: 5}
        ratings2 = self.item_ratings.get(item2, {})                # -> Get the ratings for item2, default to empty dict if not found, ex: ratings2 = {101: 3, 103: 4, 104: 2}
        
        # Find common users who rated both items
        common_users = set(ratings1.keys()) & set(ratings2.keys()) # -> Find the intersection of user IDs who rated both items, ex: common_users = {101, 104}
        
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
        vec1 = [ratings1[u] for u in common_users]                 # -> Create a vector of ratings for item1 from common users, ex: vec1 = [5, 3]
        vec2 = [ratings2[u] for u in common_users]                 # -> Create a vector of ratings for item2 from common users, ex: vec2 = [3, 2]
        
        # Compute dot product
        dot_product = sum(r1 * r2 for r1, r2 in zip(vec1, vec2))   # -> Calculate the dot product of the two vectors, ex: dot_product = 5 * 3 + 3 * 2 = 21
        
        # Compute magnitudes
        mag1 = math.sqrt(sum(r * r for r in vec1))                 # -> Calculate the magnitude of vec1, ex: mag1 = sqrt(5^2 + 3^2) = sqrt(34) = 5.83
        mag2 = math.sqrt(sum(r * r for r in vec2))                 # -> Calculate the magnitude of vec2, ex: mag2 = sqrt(3^2 + 2^2) = sqrt(13) = 3.61
        
        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0
        
        return dot_product / (mag1 * mag2)                         # -> Calculate cosine similarity as dot_product / (mag1 * mag2), ex: cosine_similarity = 21 / (5.83 * 3.61) = 0.99
    
    def _compute_pearson_similarity(self, ratings1, ratings2, common_users, item1, item2):
        """Compute Pearson correlation between two sets of ratings"""
        # Get mean ratings
        mean1 = self.item_means.get(item1, self.global_mean)       # -> Get the mean rating for item1, default to global mean if not found, ex: mean1 = (5 + 3 + 4 + 5) / 4 = 4.25
        mean2 = self.item_means.get(item2, self.global_mean)       # -> Get the mean rating for item2, default to global mean if not found, ex: mean2 = (3 + 4 + 2) / 3 = 3.0
        
        # Calculate numerator and denominators
        numerator = 0                                              # -> Initialize numerator for Pearson correlation
        denom1 = 0                                                 # -> Initialize denominator for item1
        denom2 = 0                                                 # -> Initialize denominator for item2
        
        for user in common_users:
            """
            # ratings1 = {101: 5, 102: 4, 104: 3, 106: 5}          # Movie 201
            # ratings2 = {101: 3, 103: 4, 104: 2}                  # Movie 202

            # common_users = {101, 104}                            # Users who rated both movies

            # mean1 = 4.25                                         # Mean rating for Movie 201
            # mean2 = 3.00  # Mean rating for Movie 202

            # First iteration:
                # dev1 = 5 - 4.25 = 0.75
                # dev2 = 3 - 3.0 = 0.0
                # numerator = 0.0 + (0.75 * 0.0) = 0.0
                # denom1 = 0.0 + (0.75 * 0.75) = 0.5625
                # denom2 = 0.0 + (0.0 * 0.0) = 0.0
            # Second iteration:
                # dev1 = 3 - 4.25 = -1.25
                # dev2 = 2 - 3.0 = -1.0
                # numerator = 0.0 + (-1.25 * -1.0) = 1.25
                # denom1 = 0.5625 + (-1.25 * -1.25) = 1.5625
                # denom2 = 0.0 + (-1.0 * -1.0) = 1.0
            """

            # Calculate deviation from mean
            dev1 = ratings1[user] - mean1                          # -> Calculate deviation of user rating for item1 from its mean, ex: dev1 = 5 - 4.25 = 0.75
            dev2 = ratings2[user] - mean2                          # -> Calculate deviation of user rating for item2 from its mean, ex: dev2 = 3 - 3.0 = 0.0
            
            numerator += dev1 * dev2                               # -> Update numerator with product of deviations, ex: numerator = 0.0 + (0.75 * 0.0) = 0.0
            denom1 += dev1 * dev1                                  # -> Update denominator for item1 with square of deviation, ex: denom1 = 0.0 + (0.75 * 0.75) = 0.5625
            denom2 += dev2 * dev2                                  # -> Update denominator for item2 with square of deviation, ex: denom2 = 0.0 + (0.0 * 0.0) = 0.0
        
        # Avoid division by zero
        if denom1 == 0 or denom2 == 0:
            return 0
        
        return numerator / (math.sqrt(denom1) * math.sqrt(denom2)) # -> Calculate Pearson correlation as numerator / (sqrt(denom1) * sqrt(denom2)), ex: pearson_similarity = 1.25 / (sqrt(1.5625) * sqrt(1.0)) = 857
    
    def predict(self, user_id, item_id):
        """
        Predict rating for a user-item pair using item-based CF
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating or None if prediction impossible
        """

        """
            We computed the following item-item similarity matrix:
            self.similarities = [
                [ 1.000  0.857 -1.000  1.000 -0.866]
                [ 0.857  1.000  0.000 -1.000  0.000]
                [-1.000  0.000  1.000  0.500  0.000]
                [ 1.000 -1.000  0.500  1.000  0.000]
                [-0.866  0.000  0.000  0.000  1.000]
            ]

            Example:
            User 101 has rated Movie 201 (5), Movie 202 (3), Movie 204 (4)
            User 102 has rated Movie 201 (4), Movie 203 (5), Movie 205 (3)
            User 103 has rated Movie 202 (4), Movie 203 (2), Movie 204 (3)
            User 104 has rated Movie 201 (3), Movie 202 (2), Movie 205 (5)
            User 105 has rated Movie 203 (4), Movie 204 (5), Movie 205 (4)

            We want to predict the rating for User 101 on Movie 203

            1. Check if Movie 203 is known (exists in item_id_to_index)
            2. Get index for Movie 203 (let's say it's index 2)
            3. Find items rated by User 101:
                - Movie 201 (5) -> similarity = 0.857
                - Movie 202 (3) -> similarity = 0.000
                - Movie 204 (4) -> similarity = 1.000
            4. Get top-k neighbors (let's say k=2):
                - Movie 204 (similarity = 1.000, rating = 4)
                - Movie 201 (similarity = 0.857, rating = 5)
            5. Calculate weighted average:
                weighted_sum = (1.000 * (4 - 4.00)) + (0.857 * (5 - 4.25)) = 0 + 0.6425 = 0.6425
                sim_total = 1.000 + 0.857 = 1.857
                predicted_rating = 4.00 + (0.6425 / 1.857) = 4.00 + 0.345 = 4.345
        """
        # Check if item is known
        if item_id not in self.item_id_to_index:                          # -> Check if all the item_id are in the item_id_to_index mapping
            return None
        
        # Get index for target item
        item_idx = self.item_id_to_index[item_id]                         # -> Get the index of the item_id in the item_id_to_index mapping, ex: item_idx = 0 for Movie 201
        
        # Find items rated by this user
        rated_items = []
        for i_id, user_ratings in self.item_ratings.items():              # -> Iterate through each item_id and its corresponding user ratings
            if user_id in user_ratings and i_id in self.item_id_to_index: # -> Check if user_id has rated the item and if the item_id is known
                i_idx = self.item_id_to_index[i_id]
                similarity = self.similarities[item_idx, i_idx]
                if similarity > 0:                                        # -> Only consider positive similarities
                    rating = user_ratings[user_id]
                    rated_items.append((similarity, i_id, rating))        # -> Append a tuple of (similarity, item_id, rating) for rated items, ex: rated_items = [(0.857, 201, 5), (1.000, 204, 4)]
        
        if not rated_items:
            # If no similar items found, return the item's mean rating
            return self.item_means.get(item_id, self.global_mean)         # -> If no rated items, return the item's mean rating or global mean if not available, ex: return 4.00 for Movie 203
        
        # Get top-k neighbors
        # This function returns the k largest items from the rated_items list based on the similarity value (first element of the tuple)
        # If we have fewer than k neighbors, we just use what we have
        # For this example let's say self.k = 40
        # min(self.k, len(rated_items)) -> This ensures we don't try to get more neighbors than we have rated items, ex: if k=40 and we have only 2 rated items, it will take min(40, 2) = 2
        # key=lambda x: x[0]            -> This specifies that we want to sort by the first element of the tuple (similarity)
        k_neighbors = heapq.nlargest(min(self.k, len(rated_items)), rated_items, key=lambda x: x[0]) # -> Get the top-k rated items based on similarity, ex: k_neighbors = [(1.000, 204, 4), (0.857, 201, 5)]

        # Calculate weighted average
        sim_total = sum(sim for sim, _, _ in k_neighbors)                 # -> Calculate the total similarity of the top-k neighbors, ex: sim_total = 1.000 + 0.857 = 1.857
        if sim_total == 0:
            return self.item_means.get(item_id, self.global_mean)         # -> If total similarity is zero, return the item's mean rating or global mean if not available, ex: return 4.00 for Movie 203
        
        # For item-based CF, we can use either weighted average or adjusted weighted average
        # Here we use adjusted weighted average accounting for item mean differences
        target_mean = self.item_means.get(item_id, self.global_mean)      # -> Get the mean rating for the target item, ex: target_mean = 4.00 for Movie 203
        weighted_sum = 0                                                  # -> Initialize weighted sum for the prediction
        
        for sim, i_id, rating in k_neighbors:
            # Adjust for item mean difference
            i_mean = self.item_means.get(i_id, self.global_mean)          # -> Get the mean rating for the neighbor item, ex: i_mean = 4.25 for Movie 201 and 3.00 for Movie 204
            weighted_sum += sim * (rating - i_mean)                       # -> Calculate the weighted sum of deviations from the neighbor item mean, ex: weighted_sum = (1.000 * (4 - 4.25)) + (0.857 * (5 - 4.25)) = 0 + 0.6425 = 0.6425
        
        # Predict as target item's mean plus weighted deviations
        predicted_rating = target_mean + (weighted_sum / sim_total)       # -> Calculate the predicted rating as target item's mean plus the weighted average of deviations, ex: predicted_rating = 4.00 + (0.6425 / 1.857) = 4.00 + 0.345 = 4.345
        
        # Clip to valid rating range (typically 1-5)
        predicted_rating = max(1.0, min(5.0, predicted_rating))           # -> Ensure the predicted rating is within the valid range, ex: predicted_rating = max(1.0, min(5.0, 4.345)) = 4.345
        
        return predicted_rating                                           # -> Return the predicted rating for the user-item pair, ex: return 4.345 for User 101 on Movie 203


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

    """
    | Agorithms                       | RMSE   | MAE       |
    |---------------------------------|--------|-----------|
    | Random                          | 1.4385 |    1.1478 |
    | Collaborative Filtering         |--------|-----------|
    | ItemKNN-Pearson                 | 0.9287 |    0.7109 |   
    | ItemKNN-Cosine                  | 0.9263 |    0.7103 |  
    """
    
    # Run evaluation
    print("Evaluating algorithms...")
    evaluator.Evaluate(False)
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)