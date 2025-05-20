# -*- coding: utf-8 -*-
"""
Custom Naive Bayes Algorithm Implementation without Surprise but compatible with Framework evaluation
"""
import sys
import os
# Get the absolute path to the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Libs used for PureNaiveBayes
from EvaluationFramework.MovieLens import MovieLens
import numpy as np
import math

# Libs used for AdaptedNaiveBayes
from surprise import AlgoBase, PredictionImpossible

# Libs used for Evaluation
from utils.LoadMovieLensData import LoadMovieLensData
from EvaluationFramework.Evaluator import Evaluator
from surprise import NormalPredictor
import random


class PureNaiveBayes:
    """
    A pure Python implementation of Naive Bayes for rating prediction
    """
    
    def __init__(self, rating_levels=5):
        """Initialize Naive Bayes classifier"""
        self.rating_levels = rating_levels # Number of possible rating values (e.g., 1-5)
        self.genre_priors = {}             # P(genre=1)
        self.year_mean = {}                # Mean year for each rating
        self.year_var = {}                 # Variance of year for each rating
        self.rating_counts = {}            # Count of each rating value
        self.total_ratings = 0             # Total number of ratings
        self.rating_priors = {}            # P(rating)
        self.user_avg_ratings = {}         # Average rating for each user
        self.global_mean = 0.0             # Global mean rating
    
    def fit(self, ratings_data):
        """
        Train the Naive Bayes model
        
        Args:
            ratings_data: List of (user_id, movie_id, rating) tuples
        """
        print("Loading movie features for Naive Bayes training...")
        
        # Load content features
        ml = MovieLens()
        self.genres = ml.getGenres()
        self.years = ml.getYears()
        
        # Process all ratings
        print("Processing ratings data...")
        valid_ratings = []
        self.users = set()
        
        # Filter ratings where we have movie data
        for user_id, movie_id, rating in ratings_data:
            if movie_id in self.genres and movie_id in self.years:
                valid_ratings.append((user_id, movie_id, rating))
                self.users.add(user_id)
        
        self.total_ratings = len(valid_ratings)
        print(f"Found {self.total_ratings} valid ratings from {len(self.users)} users")
        
        # Calculate user average ratings (for personalization)
        user_ratings = {}
        for user_id, movie_id, rating in valid_ratings:
            if user_id not in user_ratings:
                user_ratings[user_id] = []
            user_ratings[user_id].append(rating)
        
        for user_id, ratings in user_ratings.items():
            self.user_avg_ratings[user_id] = sum(ratings) / len(ratings)
        
        # Calculate global mean rating
        all_ratings = [rating for _, _, rating in valid_ratings]
        self.global_mean = sum(all_ratings) / len(all_ratings)
        
        # Initialize counts
        self.rating_counts = {r: 0 for r in range(1, self.rating_levels + 1)}
        
        # Initialize genre priors for each rating level
        for r in range(1, self.rating_levels + 1):
            self.genre_priors[r] = {}
        
        # Initialize year stats for each rating level
        year_values = {r: [] for r in range(1, self.rating_levels + 1)}
        
        # Count occurrences of each rating
        for _, movie_id, rating in valid_ratings:
            # Round rating to nearest integer if needed and ensure it's in range 1-5
            rating_level = max(1, min(self.rating_levels, round(rating)))
            self.rating_counts[rating_level] += 1
            
            # Add year data for this rating
            year_values[rating_level].append(self.years[movie_id])
            
            # Update genre counts for this rating
            if rating_level not in self.genre_priors:
                self.genre_priors[rating_level] = {}
                
            for i, has_genre in enumerate(self.genres[movie_id]):
                if has_genre == 1:
                    if i not in self.genre_priors[rating_level]:
                        self.genre_priors[rating_level][i] = 0
                    self.genre_priors[rating_level][i] += 1
        
        # Calculate rating priors: P(rating)
        for r in range(1, self.rating_levels + 1):
            self.rating_priors[r] = self.rating_counts[r] / self.total_ratings
            
        # Calculate P(genre=1|rating) with Laplace smoothing
        for r in range(1, self.rating_levels + 1):
            rating_count = self.rating_counts[r]
            for genre_idx in range(len(next(iter(self.genres.values())))):
                genre_count = self.genre_priors[r].get(genre_idx, 0)
                # Apply Laplace smoothing
                self.genre_priors[r][genre_idx] = (genre_count + 1) / (rating_count + 2)
        
        # Calculate year mean and variance for each rating
        for r in range(1, self.rating_levels + 1):
            if year_values[r]:
                self.year_mean[r] = np.mean(year_values[r])
                self.year_var[r] = np.var(year_values[r]) + 1.0  # Add 1 to avoid zero variance
            else:
                # If no examples, use global stats
                all_years = list(self.years.values())
                self.year_mean[r] = np.mean(all_years)
                self.year_var[r] = np.var(all_years) + 1.0
        
        print("Naive Bayes training complete!")
        return self
    
    def _gaussian_probability(self, x, mean, var):
        """Calculate Gaussian probability density function"""
        return (1 / math.sqrt(2 * math.pi * var)) * math.exp(-((x - mean) ** 2) / (2 * var))
    
    def predict(self, user_id, movie_id):
        """
        Predict rating for a user-movie pair using Naive Bayes
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating or None if prediction impossible
        """
        # Check if movie features are available
        if movie_id not in self.genres or movie_id not in self.years:
            return None
        
        # Get movie features
        genres = self.genres[movie_id]
        year = self.years[movie_id]
        
        # Calculate P(features|rating) * P(rating) for each rating level
        likelihoods = {}
        for r in range(1, self.rating_levels + 1):
            # Start with prior probability of this rating
            likelihood = math.log(self.rating_priors[r])
            
            # Add genre likelihoods
            for idx, has_genre in enumerate(genres):
                if has_genre == 1:
                    # P(genre=1|rating)
                    likelihood += math.log(self.genre_priors[r][idx])
                else:
                    # P(genre=0|rating) = 1 - P(genre=1|rating)
                    likelihood += math.log(1.0 - self.genre_priors[r][idx])
            
            # Add year likelihood using Gaussian model
            year_likelihood = self._gaussian_probability(year, self.year_mean[r], self.year_var[r])
            if year_likelihood > 0:
                likelihood += math.log(year_likelihood)
            
            likelihoods[r] = likelihood
        
        # Find rating with maximum likelihood
        max_rating = max(likelihoods, key=likelihoods.get)
        
        # Personalize prediction based on user's average rating
        if user_id in self.user_avg_ratings:
            user_bias = self.user_avg_ratings[user_id] - self.global_mean
            personalized_rating = max_rating + 0.25 * user_bias
            return max(1.0, min(5.0, personalized_rating))
        else:
            return float(max_rating)


class AdaptedNaiveBayes(AlgoBase):
    """
    Adapter class that wraps PureNaiveBayes to make it compatible with Surprise's AlgoBase
    for evaluation with the Framework
    """
    
    def __init__(self):
        """Initialize Naive Bayes classifier"""
        AlgoBase.__init__(self)
        self.pure_nb = PureNaiveBayes()
        
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
        
        # Train our pure Python Naive Bayes implementation
        self.pure_nb.fit(ratings_data)
        
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
            
            # Get prediction from pure Naive Bayes
            prediction = self.pure_nb.predict(user_id, movie_id)
            
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
    naive_bayes = AdaptedNaiveBayes()
    evaluator.AddAlgorithm(naive_bayes, "PureNaiveBayes")
    
    random_predictor = NormalPredictor()
    evaluator.AddAlgorithm(random_predictor, "Random")
    
    # Run evaluation
    print("Evaluating algorithms...")
    evaluator.Evaluate(False)
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)