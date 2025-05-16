# -*- coding: utf-8 -*-
"""
Custom Content-Based KNN Algorithm Implementation without Surprise but compatible with Framework evaluation
"""
import sys
import os
# Get the absolute path to the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Libs used for PutKNN
from EvaluationFramework.MovieLens import MovieLens
import numpy as np
import math
import heapq

# Libs used for AdaptedKNN
from surprise import AlgoBase, PredictionImpossible

# Libs used for Evaluation of AdaptedKNN
from utils.LoadMovieLensData import LoadMovieLensData
from EvaluationFramework.Evaluator import Evaluator
from surprise import NormalPredictor
import random


class PureKNN:
    """
    A pure Python implementation of content-based KNN algorithm
    """
    
    def __init__(self, k=40):
        """Initialize with k nearest neighbors"""
        self.k = k
        self.similarities = None
        self.movie_index_to_id = {}
        self.movie_id_to_index = {}
        self.user_ratings = {}  # Dictionary of user_id -> {movie_id -> rating}
        
    def fit(self, ratings_data):
        """
        Train the algorithm using pure Python (no Surprise)
        
        Args:
            ratings_data: List of (user_id, movie_id, rating) tuples
        """
        # Load content features using MovieLens helper
        ml = MovieLens()
        self.genres = ml.getGenres()
        self.years = ml.getYears()
        
        # Process ratings data
        all_movie_ids = set()
        for user_id, movie_id, rating in ratings_data:
            # Initialize user dictionary if needed
            if user_id not in self.user_ratings:
                self.user_ratings[user_id] = {}
            
            # Store rating
            self.user_ratings[user_id][movie_id] = rating
            all_movie_ids.add(movie_id)
        
        # Create movie ID mappings
        self.movie_ids = sorted(list(all_movie_ids))
        self.movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        self.movie_index_to_id = {idx: movie_id for movie_id, idx in self.movie_id_to_index.items()}
        
        # Compute similarity matrix
        n_items = len(self.movie_ids)
        self.similarities = np.zeros((n_items, n_items))
        
        print("Computing content-based similarity matrix...")
        # Track progress
        processed = 0
        total = (n_items * (n_items - 1)) // 2
        
        for i in range(n_items):
            if i % 100 == 0:
                progress = (processed / total) * 100
                print(f"Progress: {progress:.1f}% ({processed}/{total}) ({i} items out of {n_items})")
                
            movie_id_i = self.movie_index_to_id[i]
            
            for j in range(i+1, n_items):
                movie_id_j = self.movie_index_to_id[j]
                
                # Skip if either movie lacks content data
                if movie_id_i not in self.genres or movie_id_j not in self.genres or \
                   movie_id_i not in self.years or movie_id_j not in self.years:
                    continue
                
                # Compute similarity components
                genre_sim = self._compute_genre_similarity(movie_id_i, movie_id_j)
                year_sim = self._compute_year_similarity(movie_id_i, movie_id_j)
                
                # Compute combined similarity
                sim = genre_sim * year_sim
                
                # Store symmetrically
                self.similarities[i, j] = sim
                self.similarities[j, i] = sim
                
                processed += 1
        
        print("Similarity computation complete!")
        return self
    
    def _compute_genre_similarity(self, movie1, movie2):
        """Compute cosine similarity between genre vectors"""
        genres1 = self.genres[movie1]
        genres2 = self.genres[movie2]
        
        # Compute cosine similarity
        dot_product = 0
        norm1 = 0
        norm2 = 0
        
        for i in range(len(genres1)):
            dot_product += genres1[i] * genres2[i]
            norm1 += genres1[i] * genres1[i]
            norm2 += genres2[i] * genres2[i]
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))
    
    def _compute_year_similarity(self, movie1, movie2):
        """Compute similarity based on year difference"""
        year1 = self.years[movie1]
        year2 = self.years[movie2]
        
        # Use exponential decay for year difference
        diff = abs(year1 - year2)
        return math.exp(-diff / 10.0)
    
    def predict(self, user_id, movie_id):
        """
        Predict rating for a user-movie pair
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating or None if prediction impossible
        """
        # Check if we can make prediction
        if user_id not in self.user_ratings or movie_id not in self.movie_id_to_index:
            return None
            
        # Get index for target movie
        movie_idx = self.movie_id_to_index[movie_id]
        
        # Get user's ratings
        user_rated_movies = self.user_ratings[user_id]
        
        if not user_rated_movies:
            return None
            
        # Find similarities between target movie and all rated movies
        neighbors = []
        for rated_movie_id, rating in user_rated_movies.items():
            # Skip if movie not in similarity matrix
            if rated_movie_id not in self.movie_id_to_index:
                continue
                
            rated_movie_idx = self.movie_id_to_index[rated_movie_id]
            similarity = self.similarities[movie_idx, rated_movie_idx]
            
            if similarity > 0:  # Only consider positive similarities
                neighbors.append((similarity, rating))
        
        if not neighbors:
            return None
            
        # Get top-k neighbors
        k_neighbors = heapq.nlargest(min(self.k, len(neighbors)), neighbors, key=lambda x: x[0])
        
        # Calculate weighted average
        sim_total = sum(sim for sim, _ in k_neighbors)
        if sim_total == 0:
            return None
            
        weighted_sum = sum(sim * rating for sim, rating in k_neighbors)
        predicted_rating = weighted_sum / sim_total
        
        return predicted_rating


class AdaptedKNN(AlgoBase):
    """
    Adapter class that wraps PureKNN to make it compatible with Surprise's AlgoBase
    for evaluation with the Framework
    """
    
    def __init__(self, k=40):
        """Initialize with k nearest neighbors"""
        AlgoBase.__init__(self)
        self.k = k
        self.pure_knn = PureKNN(k=k)
        
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
            movie_id = int(trainset.to_raw_iid(i))  # Ensure movie_id is int
            rating = r
            
            ratings_data.append((user_id, movie_id, rating))
        
        # Train our pure Python KNN implementation
        self.pure_knn.fit(ratings_data)
        
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
            movie_id = int(self.trainset.to_raw_iid(i))
            
            # Get prediction from pure KNN
            prediction = self.pure_knn.predict(user_id, movie_id)
            
            if prediction is None:
                raise PredictionImpossible("Cannot make prediction for this user-item pair")
                
            return prediction
        
        except ValueError:
            # Handle unknown items or users
            raise PredictionImpossible(f"User or item is unknown: {u}, {i}")

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

# For testing and evaluation
if __name__ == "__main__":    
    np.random.seed(0)
    random.seed(0)
    
    # Load data
    (ml, evaluationData, rankings) = LoadMovieLensData()
    
    # Create evaluator
    evaluator = Evaluator(evaluationData, rankings)
    
    # Add algorithms
    contentKNN = AdaptedKNN(k=40)
    evaluator.AddAlgorithm(contentKNN, "PureKNN")
    
    random_predictor = NormalPredictor()
    evaluator.AddAlgorithm(random_predictor, "Random")
    
    # Run evaluation
    evaluator.Evaluate(False)
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)