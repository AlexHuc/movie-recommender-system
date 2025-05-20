# -*- coding: utf-8 -*-
"""
Custom TF-IDF (Term Frequency - Inverse Document Frequency) + Cosine Similarity Algorithm Implementation without Surprise
but compatible with Framework evaluation
"""
import sys
import os
# Get the absolute path to the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Libs used for PureTFIDF
import numpy as np
import pandas as pd
import math
import re
import heapq
from collections import Counter

# Libs used for AdaptedTFIDF
from surprise import AlgoBase, PredictionImpossible

# Libs used for Evaluation
from utils.LoadMovieLensData import LoadMovieLensData
from EvaluationFramework.Evaluator import Evaluator
from surprise import NormalPredictor
import random


class PureTFIDF:
    """
    A pure Python implementation of TF-IDF + Cosine Similarity for movie recommendations
    """
    
    def __init__(self, k=40):
        """Initialize with k nearest neighbors"""
        self.k = k
        self.similarities = None
        self.movie_index_to_id = {}
        self.movie_id_to_index = {}
        self.user_ratings = {}      # Dictionary of user_id -> {movie_id -> rating}
        self.tfidf_vectors = {}     # Dictionary of movie_id -> tfidf vector
        self.vocabulary = []        # List of terms in the vocabulary
        self.movie_documents = {}   # Dictionary of movie_id -> preprocessed document
        
    def _preprocess_text(self, text):
        """Clean and tokenize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize and remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 
                      'when', 'where', 'how', 'who', 'which', 'this', 'that', 'to', 'of', 
                      'in', 'for', 'with', 'on', 'by', 'about', 'at', 'from'}
        
        tokens = text.split()
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        
        return tokens
    
    def fit(self, ratings_data, movies_file, tags_file):
        """
        Train the algorithm using TF-IDF and cosine similarity
        
        Args:
            ratings_data: List of (user_id, movie_id, rating) tuples
            movies_file: Path to movies.csv
            tags_file: Path to tags.csv
        """
        print("Loading movies and tags data...")
        
        # Load movies data
        movies_df = pd.read_csv(movies_file)
        
        # Load tags data
        tags_df = pd.read_csv(tags_file)
        
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
        
        print(f"Building document corpus for {len(self.movie_ids)} movies...")
        
        # Create document for each movie (combining title, genres, and tags)
        for movie_id in self.movie_ids:
            # Get movie title and genres
            movie_info = movies_df[movies_df['movieId'] == movie_id]
            if len(movie_info) == 0:
                continue
                
            title = movie_info.iloc[0]['title']
            genres = movie_info.iloc[0]['genres'].replace('|', ' ')
            
            # Extract year from title if present
            year_match = re.search(r'\((\d{4})\)', title)
            year = ""
            if year_match:
                year = year_match.group(1)
                # Remove year from title for processing
                title = re.sub(r'\(\d{4}\)', '', title).strip()
            
            # Get tags for this movie
            movie_tags = tags_df[tags_df['movieId'] == movie_id]
            tags_text = " ".join(movie_tags['tag'].values) if len(movie_tags) > 0 else ""
            
            # Combine all text
            document = f"{title} {genres} {tags_text} {year}"
            
            # Preprocess the document
            tokens = self._preprocess_text(document)
            
            # Store preprocessed document
            self.movie_documents[movie_id] = tokens
        
        # Build vocabulary from all tokens
        all_tokens = []
        for tokens in self.movie_documents.values():
            all_tokens.extend(tokens)
        
        # Keep only tokens that appear in at least 2 documents to reduce noise
        token_counts = Counter(all_tokens)
        self.vocabulary = [token for token, count in token_counts.items() if count >= 2]
        
        print(f"Vocabulary size: {len(self.vocabulary)}")
        
        # Build term-document matrix
        print("Computing TF-IDF vectors...")
        
        # Term frequency for each document
        tf = {}
        for movie_id, tokens in self.movie_documents.items():
            tf[movie_id] = {}
            token_counts = Counter(tokens)
            doc_length = len(tokens)
            
            for token in token_counts:
                if token in self.vocabulary:
                    tf[movie_id][token] = token_counts[token] / doc_length
        
        # Inverse document frequency
        idf = {}
        num_docs = len(self.movie_documents)
        
        for term in self.vocabulary:
            doc_count = sum(1 for movie_id in self.movie_documents if term in self.movie_documents[movie_id])
            idf[term] = math.log(num_docs / (1 + doc_count))
        
        # Compute TF-IDF vectors
        for movie_id in self.movie_documents:
            self.tfidf_vectors[movie_id] = {}
            for term in self.vocabulary:
                tf_value = tf[movie_id].get(term, 0)
                self.tfidf_vectors[movie_id][term] = tf_value * idf[term]
        
        # Compute similarity matrix
        print("Computing TF-IDF similarity matrix...")
        n_items = len(self.movie_ids)
        self.similarities = np.zeros((n_items, n_items))
        
        # Track progress
        processed = 0
        total = (n_items * (n_items - 1)) // 2
        
        for i in range(n_items):
            movie_id_i = self.movie_index_to_id[i]
            
            # Skip movies without TF-IDF vectors
            if movie_id_i not in self.tfidf_vectors:
                continue
                
            for j in range(i+1, n_items):
                movie_id_j = self.movie_index_to_id[j]
                
                # Skip if either movie has no TF-IDF vector
                if movie_id_j not in self.tfidf_vectors:
                    continue
                
                # Compute cosine similarity
                similarity = self._compute_cosine_similarity(
                    self.tfidf_vectors[movie_id_i], 
                    self.tfidf_vectors[movie_id_j]
                )
                
                # Store similarity
                self.similarities[i, j] = similarity
                self.similarities[j, i] = similarity
                
                processed += 1
                if processed % 10000 == 0:
                    progress = (processed / total) * 100
                    print(f"Progress: {progress:.1f}% ({processed}/{total})")
        
        print("Similarity computation complete!")
        return self
    
    def _compute_cosine_similarity(self, vec1, vec2):
        """
        Compute cosine similarity between two sparse vectors represented as dictionaries
        """
        # Find common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        # Compute dot product for common terms
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # Compute magnitudes
        magnitude1 = math.sqrt(sum(value**2 for value in vec1.values()))
        magnitude2 = math.sqrt(sum(value**2 for value in vec2.values()))
        
        # Compute similarity
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
            
        return dot_product / (magnitude1 * magnitude2)
    
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
    
    def get_similar_movies(self, movie_id, n=10):
        """
        Get n most similar movies to the given movie
        
        Args:
            movie_id: Movie ID
            n: Number of similar movies to return
            
        Returns:
            List of (movie_id, similarity) tuples
        """
        if movie_id not in self.movie_id_to_index:
            return []
            
        movie_idx = self.movie_id_to_index[movie_id]
        
        # Get similarities to all other movies
        similarities = [(self.movie_index_to_id[i], self.similarities[movie_idx, i]) 
                       for i in range(len(self.movie_ids))
                       if i != movie_idx and self.similarities[movie_idx, i] > 0]
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top n
        return similarities[:n]


class AdaptedTFIDF(AlgoBase):
    """
    Adapter class that wraps PureTFIDF to make it compatible with Surprise's AlgoBase
    for evaluation with the Framework
    """
    
    def __init__(self, k=40, movies_file=None, tags_file=None):
        """Initialize with k nearest neighbors and file paths"""
        AlgoBase.__init__(self)
        self.k = k
        self.movies_file = movies_file or 'movie-recommender-system/ml-latest-small/movies.csv'
        self.tags_file = tags_file or 'movie-recommender-system/ml-latest-small/tags.csv'
        self.pure_tfidf = PureTFIDF(k=k)
        
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
        
        # Train our pure Python TF-IDF implementation
        self.pure_tfidf.fit(ratings_data, self.movies_file, self.tags_file)
        
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
            
            # Get prediction from pure TF-IDF
            prediction = self.pure_tfidf.predict(user_id, movie_id)
            
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
    tfidf_recommender = AdaptedTFIDF(k=40)
    evaluator.AddAlgorithm(tfidf_recommender, "TF-IDF")
    
    random_predictor = NormalPredictor()
    evaluator.AddAlgorithm(random_predictor, "Random")
    
    # Run evaluation
    print("Evaluating algorithms...")
    evaluator.Evaluate(False)
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)