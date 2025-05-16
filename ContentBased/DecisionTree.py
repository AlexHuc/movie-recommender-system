# -*- coding: utf-8 -*-
"""
Custom Decision Trees Algorithm Implementation without Surprise but compatible with Framework evaluation
"""
import sys
import os
# Get the absolute path to the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Libs used for PureDecisionTree
from EvaluationFramework.MovieLens import MovieLens
import numpy as np
import random

# Libs used for AdaptedDecisionTree
from surprise import AlgoBase, PredictionImpossible

# Libs used for Evaluation of AdaptedDecisionTree
from utils.LoadMovieLensData import LoadMovieLensData
from EvaluationFramework.Evaluator import Evaluator
from surprise import NormalPredictor


class Node:
    """Node in the decision tree"""
    def __init__(self):
        self.left = None
        self.right = None
        self.feature_idx = None
        self.threshold = None
        self.value = None  # For leaf nodes (predicted value)
        self.is_leaf = False


class PureDecisionTree:
    """
    A pure Python implementation of Decision Trees for rating prediction
    """
    
    def __init__(self, max_depth=8, min_samples_split=5):
        """Initialize Decision Tree parameters"""
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.user_trees = {}       # Dictionary to store decision trees for each user
        self.movie_features = {}   # Dictionary to store feature vectors for each movie
        self.user_avg_ratings = {} # Average rating for each user (for fallback)
        self.global_mean = 0.0     # Global mean rating (for fallback)
    
    def fit(self, ratings_data):
        """
        Train the Decision Tree models
        
        Args:
            ratings_data: List of (user_id, movie_id, rating) tuples
        """
        # Load content features using MovieLens helper
        print("Loading movie features for Decision Tree training...")
        ml = MovieLens()
        genres = ml.getGenres()
        years = ml.getYears()
        
        # Build feature vectors for all movies
        print("Creating movie feature vectors...")
        all_movie_ids = set(movie_id for _, movie_id, _ in ratings_data)
        
        for movie_id in all_movie_ids:
            # Skip movies without genre or year data
            if movie_id not in genres or movie_id not in years:
                continue
            
            # Create feature vector: genres + normalized year
            feature_vector = genres[movie_id] + [years[movie_id] / 2020.0]  # Simple normalization for year
            self.movie_features[movie_id] = np.array(feature_vector)
        
        print(f"Created feature vectors for {len(self.movie_features)} movies")
        
        # Group ratings by user
        user_ratings = {}
        all_ratings = []
        for user_id, movie_id, rating in ratings_data:
            if user_id not in user_ratings:
                user_ratings[user_id] = []
            
            if movie_id in self.movie_features:  # Only include movies with features
                user_ratings[user_id].append((movie_id, rating))
                all_ratings.append(rating)
        
        # Calculate global mean rating for fallback
        self.global_mean = sum(all_ratings) / len(all_ratings) if all_ratings else 3.0
        
        # Calculate user average ratings for personalization
        for user_id, ratings in user_ratings.items():
            if ratings:
                self.user_avg_ratings[user_id] = sum(r for _, r in ratings) / len(ratings)
        
        # Train a decision tree for each user
        print("Training decision trees for individual users...")
        min_ratings = 20  # Minimum ratings needed for a user to get a personal tree
        
        # Count eligible users
        eligible_users = sum(1 for ratings in user_ratings.values() if len(ratings) >= min_ratings)
        print(f"Training trees for {eligible_users} users with at least {min_ratings} ratings")
        
        users_trained = 0
        for user_id, ratings in user_ratings.items():
            if len(ratings) < min_ratings:
                continue
            
            # Extract feature vectors and ratings for this user
            X = np.array([self.movie_features[m] for m, _ in ratings if m in self.movie_features])
            y = np.array([r for m, r in ratings if m in self.movie_features])
            
            if len(X) < min_ratings:
                continue
            
            # Train decision tree
            self.user_trees[user_id] = self._build_tree(X, y, depth=0)
            
            users_trained += 1
            if users_trained % 10 == 0:
                progress = (users_trained / eligible_users) * 100
                print(f"Progress: {progress:.1f}% (Trained trees for {users_trained} users out of {eligible_users})")
        
        print(f"Decision Tree training complete! Models created for {len(self.user_trees)} users")
        return self
    
    def _calculate_variance(self, y):
        """Calculate variance of target values"""
        if len(y) <= 1:
            return 0
        return np.var(y)
    
    def _best_split(self, X, y):
        """Find the best feature and threshold to split the data"""
        m, n = X.shape
        if m <= self.min_samples_split:
            return None, None
        
        # Calculate current variance
        parent_var = self._calculate_variance(y)
        
        # Initialize variables for best split
        best_feat, best_thresh = None, None
        best_var_reduction = 0
        
        # Try all features and possible thresholds
        for feat in range(n):
            # Get unique values for this feature as potential thresholds
            # For binary features (genres), we only need one threshold: 0.5
            thresholds = [0.5] if feat < n-1 else np.linspace(min(X[:, feat]), max(X[:, feat]), num=10)
            
            for threshold in thresholds:
                # Split data based on threshold
                left_idx = X[:, feat] <= threshold
                right_idx = ~left_idx
                
                # Skip if split doesn't divide data
                if sum(left_idx) == 0 or sum(right_idx) == 0:
                    continue
                
                # Calculate weighted average variance after split
                left_var = self._calculate_variance(y[left_idx])
                right_var = self._calculate_variance(y[right_idx])
                
                # Calculate variance reduction (information gain)
                n_left, n_right = sum(left_idx), sum(right_idx)
                var_reduction = parent_var - (n_left * left_var + n_right * right_var) / m
                
                # Update best split if this one is better
                if var_reduction > best_var_reduction:
                    best_var_reduction = var_reduction
                    best_feat = feat
                    best_thresh = threshold
        
        return best_feat, best_thresh
    
    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree"""
        # Create a node for this split
        node = Node()
        
        # Check if we should make this a leaf node
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) <= self.min_samples_split:
            node.is_leaf = True
            node.value = np.mean(y)
            return node
        
        # Find best feature and threshold for split
        best_feat, best_thresh = self._best_split(X, y)
        
        # If no good split found, make a leaf node
        if best_feat is None:
            node.is_leaf = True
            node.value = np.mean(y)
            return node
        
        # Split the data
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = ~left_idx
        
        # Save the split information
        node.feature_idx = best_feat
        node.threshold = best_thresh
        
        # Recursively build left and right subtrees
        node.left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        node.right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        
        return node
    
    def _predict_tree(self, tree, features):
        """Traverse tree to make prediction"""
        # If leaf node, return the value
        if tree.is_leaf:
            return tree.value
        
        # Traverse left or right based on feature value
        if features[tree.feature_idx] <= tree.threshold:
            return self._predict_tree(tree.left, features)
        else:
            return self._predict_tree(tree.right, features)
    
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
        features = self.movie_features[movie_id]
        
        # If user has a tree, use it for prediction
        if user_id in self.user_trees:
            tree = self.user_trees[user_id]
            prediction = self._predict_tree(tree, features)
            
            # Adjust prediction to valid range
            prediction = max(1.0, min(5.0, prediction))
            return prediction
        
        # Fallback to user average or global mean
        if user_id in self.user_avg_ratings:
            return self.user_avg_ratings[user_id]
        else:
            return self.global_mean


class AdaptedDecisionTree(AlgoBase):
    """
    Adapter class that wraps PureDecisionTree to make it compatible with Surprise's AlgoBase
    for evaluation with the Framework
    """
    
    def __init__(self, max_depth=8, min_samples_split=5):
        """Initialize Decision Tree parameters"""
        AlgoBase.__init__(self)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.pure_dt = PureDecisionTree(max_depth=max_depth, min_samples_split=min_samples_split)
        
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
        
        # Train our pure Python Decision Tree implementation
        self.pure_dt.fit(ratings_data)
        
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
            
            # Get prediction from pure Decision Tree
            prediction = self.pure_dt.predict(user_id, movie_id)
            
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
    dt_recommender = AdaptedDecisionTree(max_depth=8, min_samples_split=5)
    evaluator.AddAlgorithm(dt_recommender, "DecisionTree")
    
    random_predictor = NormalPredictor()
    evaluator.AddAlgorithm(random_predictor, "Random")
    
    # Run evaluation
    print("Evaluating algorithms...")
    evaluator.Evaluate(False)
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)