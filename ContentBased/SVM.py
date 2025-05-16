# -*- coding: utf-8 -*-
"""
Custom SVM-based Recommendation Algorithm Implementation from scratch
"""
import sys
import os
# Get the absolute path to the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Libs used for PutSVM
from EvaluationFramework.MovieLens import MovieLens
import numpy as np
import random

# Libs used for AdaptedSVM
from surprise import AlgoBase, PredictionImpossible

# Libs used for Evaluation of AdaptedSVM
from utils.LoadMovieLensData import LoadMovieLensData
from EvaluationFramework.Evaluator import Evaluator
from surprise import NormalPredictor
import random


class PureSVM:
    """
    A pure Python implementation of Support Vector Regression for recommendation
    """
    
    def __init__(self, C=1.0, epsilon=0.1, learning_rate=0.001, max_iterations=1000):
        """Initialize SVM parameters"""
        self.C = C  # Regularization parameter
        self.epsilon = epsilon  # Epsilon in the epsilon-SVM model
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.user_models = {}  # Dictionary to store model weights for each user
        self.movie_features = {}  # Dictionary to store movie features
        
    def _normalize_features(self, features):
        """Simple feature normalization to [0,1] range"""
        min_vals = np.min(features, axis=0)
        max_vals = np.max(features, axis=0)
        range_vals = max_vals - min_vals
        
        # Handle zero range (constant features)
        range_vals[range_vals == 0] = 1
        
        normalized = (features - min_vals) / range_vals
        return normalized, min_vals, range_vals
    
    def _compute_gradient(self, X, y, weights, bias, C, epsilon):
        """Compute gradient for SVM optimization"""
        n_samples = X.shape[0]
        
        # Initialize gradients
        weight_grad = np.zeros_like(weights)
        bias_grad = 0
        
        for i in range(n_samples):
            prediction = np.dot(X[i], weights) + bias
            error = prediction - y[i]
            
            # SVM loss function has a flat region within epsilon
            if abs(error) <= epsilon:
                # No gradient contribution from samples predicted within epsilon
                continue
            
            # Sign of the error determines the gradient direction
            sign = 1 if error > 0 else -1
            
            # Update weight gradient
            weight_grad += sign * X[i]
            
            # Update bias gradient
            bias_grad += sign
        
        # Apply regularization to weights (L2 regularization)
        weight_grad += 2 * C * weights
        
        # Normalize gradients by sample count
        weight_grad /= n_samples
        bias_grad /= n_samples
        
        return weight_grad, bias_grad
    
    def _stochastic_gradient_descent(self, X, y, C, epsilon, learning_rate, max_iterations):
        """Optimize SVM parameters using stochastic gradient descent"""
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        weights = np.zeros(n_features)
        bias = 0
        
        # Normalize features
        X_normalized, min_vals, range_vals = self._normalize_features(X)
        
        # For tracking convergence
        prev_loss = float('inf')
        
        # SGD iterations
        for iteration in range(max_iterations):
            # Shuffle the data
            indices = list(range(n_samples))
            random.shuffle(indices)
            
            # Perform updates using mini-batches
            batch_size = min(50, n_samples)
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                X_batch = X_normalized[batch_indices]
                y_batch = y[batch_indices]
                
                # Compute gradients
                weight_grad, bias_grad = self._compute_gradient(
                    X_batch, y_batch, weights, bias, C, epsilon)
                
                # Update parameters
                weights -= learning_rate * weight_grad
                bias -= learning_rate * bias_grad
            
            # Compute loss for convergence check (every 50 iterations to save time)
            if iteration % 50 == 0:
                predictions = np.dot(X_normalized, weights) + bias
                errors = predictions - y
                
                # SVM loss: epsilon-insensitive loss + regularization
                loss = 0
                for error in errors:
                    loss += max(0, abs(error) - epsilon)
                
                # Add regularization term
                loss += C * np.sum(weights**2)
                
                # Check for convergence
                if abs(prev_loss - loss) < 1e-4:
                    break
                
                prev_loss = loss
        
        return weights, bias, min_vals, range_vals
    
    def fit(self, ratings_data):
        """
        Train SVM models for each user
        
        Args:
            ratings_data: List of (user_id, movie_id, rating) tuples
        """
        # Load content features using MovieLens helper
        ml = MovieLens()
        genres = ml.getGenres()
        years = ml.getYears()
        
        print("Preparing movie features for SVM training...")
        
        # First, build feature vectors for all movies
        all_movie_ids = set(movie_id for _, movie_id, _ in ratings_data)
        
        # Process features for each movie
        for movie_id in all_movie_ids:
            # Skip movies without genre or year data
            if movie_id not in genres or movie_id not in years:
                continue
            
            # Get genre vector
            genre_vector = genres[movie_id]
            
            # Get year and convert to feature
            year = years[movie_id]
            
            # Combine features
            feature_vector = genre_vector + [year]
            
            # Store feature vector
            self.movie_features[movie_id] = np.array(feature_vector)
        
        print(f"Created feature vectors for {len(self.movie_features)} movies")
        
        # Group ratings by user
        user_ratings = {}
        for user_id, movie_id, rating in ratings_data:
            if user_id not in user_ratings:
                user_ratings[user_id] = []
            user_ratings[user_id].append((movie_id, rating))
        
        # Train an SVM model for each user with enough ratings
        min_ratings = 10  # Minimum number of ratings needed to train a model
        print(f"Training individual SVM models for each user...")
        
        # Count eligible users after collecting all ratings
        eligible_users = 0
        for user_id, ratings in user_ratings.items():
            valid_ratings = [(m, r) for m, r in ratings if m in self.movie_features]
            if len(valid_ratings) >= min_ratings:
                eligible_users += 1
    
        users_trained = 0
        for user_id, ratings in user_ratings.items():
            # Filter to only include movies with feature vectors
            valid_ratings = [(m, r) for m, r in ratings if m in self.movie_features]
            
            if len(valid_ratings) < min_ratings:
                continue  # Skip users with too few ratings
            
            # Extract feature vectors and ratings
            X = np.array([self.movie_features[m] for m, _ in valid_ratings])
            y = np.array([r for _, r in valid_ratings])
            
            # Train SVM model using our pure SGD implementation
            # print(f"Training model for user {user_id} with {len(valid_ratings)} ratings")
            weights, bias, min_vals, range_vals = self._stochastic_gradient_descent(
                X, y, self.C, self.epsilon, self.learning_rate, self.max_iterations)
            
            # Store the model parameters
            self.user_models[user_id] = {
                'weights': weights,
                'bias': bias,
                'min_vals': min_vals,
                'range_vals': range_vals
            }
            
            users_trained += 1
            if users_trained % 10 == 0:
                progress = (users_trained / eligible_users) * 100
                print(f"Progress: {progress:.1f}% (Trained models for {users_trained} users out of {eligible_users})")
        
        print(f"SVM training complete! Models trained for {len(self.user_models)} users")
        return self
    
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
        if user_id not in self.user_models or movie_id not in self.movie_features:
            return None
        
        # Get user's model
        model = self.user_models[user_id]
        weights = model['weights']
        bias = model['bias']
        min_vals = model['min_vals']
        range_vals = model['range_vals']
        
        # Get movie features and normalize
        features = self.movie_features[movie_id]
        features_normalized = (features - min_vals) / range_vals
        
        # Make prediction
        prediction = np.dot(features_normalized, weights) + bias
        
        # Clip prediction to valid rating range (1-5)
        prediction = max(1.0, min(5.0, prediction))
        
        return prediction


class AdaptedSVM(AlgoBase):
    """
    Adapter class that wraps PureSVM to make it compatible with Surprise's AlgoBase
    for evaluation with the Framework
    """
    
    def __init__(self, C=1.0, epsilon=0.1, learning_rate=0.001, max_iterations=1000):
        """Initialize SVM parameters"""
        AlgoBase.__init__(self)
        self.C = C
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.pure_svm = PureSVM(C=C, epsilon=epsilon, 
                               learning_rate=learning_rate, 
                               max_iterations=max_iterations)
        
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
        
        # Train our pure Python SVM implementation
        self.pure_svm.fit(ratings_data)
        
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
            
            # Get prediction from pure SVM
            prediction = self.pure_svm.predict(user_id, movie_id)
            
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
    svmRecommender = AdaptedSVM(C=1.0, epsilon=0.1, learning_rate=0.01, max_iterations=500)
    evaluator.AddAlgorithm(svmRecommender, "PureSVM")
    
    random_predictor = NormalPredictor()
    evaluator.AddAlgorithm(random_predictor, "Random")
    
    # Run evaluation
    print("Evaluating algorithms...")
    evaluator.Evaluate(False)
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)