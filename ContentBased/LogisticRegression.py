# -*- coding: utf-8 -*-
"""
Custom Logistic Regression Algorithm Implementation without Surprise but compatible with Framework evaluation
"""
import sys
import os
# Get the absolute path to the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Libs used for PureLogisticRegression
from EvaluationFramework.MovieLens import MovieLens
import numpy as np

# Libs used for AdaptedLogisticRegression
from surprise import AlgoBase, PredictionImpossible

# Libs used for Evaluation
from utils.LoadMovieLensData import LoadMovieLensData
from EvaluationFramework.Evaluator import Evaluator
from surprise import NormalPredictor
import random


class PureLogisticRegression:
    """
    A pure Python implementation of logistic regression for rating prediction
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, rating_threshold=3.5):
        """Initialize logistic regression parameters"""
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.rating_threshold = rating_threshold  # Threshold to classify ratings as "liked" or "not liked"
        self.user_models = {}  # Dictionary to store model weights for each user
        self.movie_features = {}  # Dictionary to store normalized feature vectors for each movie
        self.feature_means = None  # Feature means for normalization
        self.feature_stds = None   # Feature standard deviations for normalization
        self.user_avg_ratings = {}  # Average rating for each user (for prediction)
        self.global_mean = 0.0     # Global mean rating (fallback)
        
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip to avoid overflow
        z = np.clip(z, -20, 20)
        return 1.0 / (1.0 + np.exp(-z))
    
    def _normalize_features(self, features):
        """Normalize features to have zero mean and unit variance"""
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0) + 1e-8  # Avoid division by zero
        normalized = (features - means) / stds
        return normalized, means, stds
    
    def fit(self, ratings_data):
        """
        Train logistic regression models for each user
        
        Args:
            ratings_data: List of (user_id, movie_id, rating) tuples
        """
        print("Loading movie features for logistic regression training...")
        
        # Load content features
        ml = MovieLens()
        genres = ml.getGenres()
        years = ml.getYears()
        
        # Collect all movie IDs from ratings
        all_movie_ids = set(movie_id for _, movie_id, _ in ratings_data)
        
        # Create feature vectors for movies
        raw_features = []
        feature_movie_ids = []
        
        for movie_id in all_movie_ids:
            if movie_id in genres and movie_id in years:
                # Create feature vector: genres + year
                feature_vector = np.array(genres[movie_id] + [years[movie_id]])
                raw_features.append(feature_vector)
                feature_movie_ids.append(movie_id)
        
        if not raw_features:
            print("No valid movie features found!")
            return self
        
        # Convert to numpy array for processing
        raw_features = np.array(raw_features)
        
        # Normalize all features together
        normalized_features, self.feature_means, self.feature_stds = self._normalize_features(raw_features)
        
        # Create normalized movie feature dictionary
        for i, movie_id in enumerate(feature_movie_ids):
            self.movie_features[movie_id] = normalized_features[i]
            
        print(f"Created normalized feature vectors for {len(self.movie_features)} movies")
        
        # Group ratings by user
        user_ratings = {}
        all_ratings = []
        
        for user_id, movie_id, rating in ratings_data:
            if user_id not in user_ratings:
                user_ratings[user_id] = []
            
            if movie_id in self.movie_features:
                user_ratings[user_id].append((movie_id, rating))
                all_ratings.append(rating)
        
        # Calculate global mean
        self.global_mean = sum(all_ratings) / len(all_ratings) if all_ratings else 3.0
        
        # Calculate user mean ratings
        for user_id, ratings in user_ratings.items():
            if ratings:
                self.user_avg_ratings[user_id] = sum(r for _, r in ratings) / len(ratings)
        
        # Train a model for each user with enough ratings
        min_ratings = 20  # Minimum number of ratings to train a model
        print(f"Training logistic regression models for users...")
        
        # Count eligible users
        eligible_users = sum(1 for ratings in user_ratings.values() if len(ratings) >= min_ratings)
        print(f"Found {eligible_users} users with at least {min_ratings} ratings")
        
        users_trained = 0
        for user_id, ratings in user_ratings.items():
            if len(ratings) < min_ratings:
                continue
                
            # Create feature matrix and target vector
            X = []
            y = []
            
            for movie_id, rating in ratings:
                if movie_id in self.movie_features:
                    X.append(self.movie_features[movie_id])
                    # Convert rating to binary: 1 if liked (rating > threshold), 0 otherwise
                    y.append(1.0 if rating > self.rating_threshold else 0.0)
            
            if len(X) < min_ratings:
                continue
                
            X = np.array(X)
            y = np.array(y)
            
            # Add bias term (column of ones)
            X_with_bias = np.column_stack((np.ones(X.shape[0]), X))
            
            # Train logistic regression model
            weights = self._train_logistic_regression(X_with_bias, y)
            
            # Store model weights
            self.user_models[user_id] = weights
            
            users_trained += 1
            if users_trained % 10 == 0:
                progress = (users_trained / eligible_users) * 100
                print(f"Progress: {progress:.1f}% (Trained models for {users_trained} users out of {eligible_users})")
        
        print(f"Logistic Regression training complete! Models created for {len(self.user_models)} users")
        return self
    
    def _train_logistic_regression(self, X, y):
        """
        Train logistic regression using gradient descent
        
        Args:
            X: Feature matrix with bias term
            y: Binary target vector (1 = liked, 0 = not liked)
            
        Returns:
            weights: Trained model weights
        """
        n_samples, n_features = X.shape
        
        # Initialize weights
        weights = np.zeros(n_features)
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Compute predictions
            z = np.dot(X, weights)
            predictions = self._sigmoid(z)
            
            # Compute gradient
            gradient = np.dot(X.T, predictions - y) / n_samples
            
            # Update weights
            weights -= self.learning_rate * gradient
            
            # Early stopping if gradient is small
            if np.all(np.abs(gradient) < 1e-5):
                break
        
        return weights
    
    def predict(self, user_id, movie_id):
        """
        Predict rating for a user-movie pair
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating or None if prediction impossible
        """
        # Check if movie has features
        if movie_id not in self.movie_features:
            return None
            
        # Get movie features
        movie_features = self.movie_features[movie_id]
        
        # If user has a model, use it to predict
        if user_id in self.user_models:
            # Add bias term
            features_with_bias = np.hstack(([1.0], movie_features))
            
            # Get weights
            weights = self.user_models[user_id]
            
            # Compute probability
            z = np.dot(features_with_bias, weights)
            probability = self._sigmoid(z)
            
            # Convert probability to rating scale
            # 0.5 probability corresponds to the threshold rating
            # Scale to full rating range
            if probability >= 0.5:
                # Map [0.5, 1.0] to [threshold, 5.0]
                rating = self.rating_threshold + (5.0 - self.rating_threshold) * (probability - 0.5) * 2
            else:
                # Map [0.0, 0.5] to [1.0, threshold]
                rating = 1.0 + (self.rating_threshold - 1.0) * probability * 2
                
            return rating
        
        # Fallback to user average or global mean
        if user_id in self.user_avg_ratings:
            return self.user_avg_ratings[user_id]
        else:
            return self.global_mean


class AdaptedLogisticRegression(AlgoBase):
    """
    Adapter class that wraps PureLogisticRegression to make it compatible with Surprise's AlgoBase
    for evaluation with the Framework
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, rating_threshold=3.5):
        """Initialize logistic regression parameters"""
        AlgoBase.__init__(self)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.rating_threshold = rating_threshold
        self.pure_lr = PureLogisticRegression(
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            rating_threshold=rating_threshold
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
            movie_id = int(trainset.to_raw_iid(i))  # Ensure movie_id is int
            rating = r
            
            ratings_data.append((user_id, movie_id, rating))
        
        # Train our pure Python Logistic Regression implementation
        self.pure_lr.fit(ratings_data)
        
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
            
            # Get prediction from pure Logistic Regression
            prediction = self.pure_lr.predict(user_id, movie_id)
            
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
    lr_recommender = AdaptedLogisticRegression(learning_rate=0.05, max_iterations=500, rating_threshold=3.5)
    evaluator.AddAlgorithm(lr_recommender, "LogisticRegression")
    
    random_predictor = NormalPredictor()
    evaluator.AddAlgorithm(random_predictor, "Random")
    
    # Run evaluation
    print("Evaluating algorithms...")
    evaluator.Evaluate(False)
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)