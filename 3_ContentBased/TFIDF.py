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
        self.k = k                                # -> Number of nearest neighbors to use for rating prediction
        self.similarities = None                  # -> Matrix to store similarities between all movie pairs
        self.movie_index_to_id = {}               # -> Dictionary mapping from index to movie_id
        self.movie_id_to_index = {}               # -> Dictionary mapping from movie_id to index
        self.user_ratings = {}                    # -> Dictionary of user_id -> {movie_id -> rating}
        self.tfidf_vectors = {}                   # -> Dictionary of movie_id -> tfidf vector
        self.vocabulary = []                      # -> List of terms in the vocabulary
        self.movie_documents = {}                 # -> Dictionary of movie_id -> preprocessed document
        
    def _preprocess_text(self, text):
        """
        Clean and tokenize text
        
        Args:
            text: Raw text string to preprocess
            
        Returns:
            list: List of preprocessed tokens
        """
        """
        Example:
            Input text: "The Shawshank Redemption (1994) Drama|Crime"
            
            Step 1: Convert to lowercase
                "the shawshank redemption (1994) drama|crime"
                
            Step 2: Remove punctuation
                "the shawshank redemption  1994  drama crime"
                
            Step 3: Remove numbers
                "the shawshank redemption   drama crime"
                
            Step 4: Tokenize and remove stop words
                ["shawshank", "redemption", "drama", "crime"]
                
            Output: ["shawshank", "redemption", "drama", "crime"]
        """
        # Convert to lowercase
        text = text.lower()                                                 # -> "the shawshank redemption (1994) drama|crime"
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)                                # -> "the shawshank redemption  1994  drama crime"
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)                                     # -> "the shawshank redemption   drama crime"
        
        # Tokenize and remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 
                      'when', 'where', 'how', 'who', 'which', 'this', 'that', 'to', 'of', 
                      'in', 'for', 'with', 'on', 'by', 'about', 'at', 'from'}
        
        tokens = text.split()                                               # -> ["the", "shawshank", "redemption", "drama", "crime"]
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]  # -> ["shawshank", "redemption", "drama", "crime"]
        
        return tokens
    
    def fit(self, ratings_data, movies_file, tags_file):
        """
        Train the algorithm using TF-IDF and cosine similarity
        
        Args:
            ratings_data: List of (user_id, movie_id, rating) tuples
            movies_file: Path to movies.csv
            tags_file: Path to tags.csv
        """
        """
        Example with a small dataset:
        
        ratings_data = [
            (1, 1, 5),  # User 1 rated Movie 1 with 5 stars
            (1, 2, 3),  # User 1 rated Movie 2 with 3 stars
            (2, 1, 4),  # User 2 rated Movie 1 with 4 stars
            (2, 3, 5)   # User 2 rated Movie 3 with 5 stars
        ]
        
        From movies.csv:
        Movie 1: "Toy Story (1995)" with genres "Animation|Adventure|Comedy"
        Movie 2: "Jumanji (1995)" with genres "Action|Adventure|Fantasy"
        Movie 3: "Heat (1995)" with genres "Action|Crime|Thriller"
        
        From tags.csv:
        Movie 1 has tags: "pixar", "animation", "children"
        Movie 2 has tags: "fantasy", "board game"
        Movie 3 has tags: "crime", "al pacino"
        
        After preprocessing:
        Movie 1 document: ["toy", "story", "animation", "adventure", "comedy", "pixar", "children"]
        Movie 2 document: ["jumanji", "action", "adventure", "fantasy", "board", "game"]
        Movie 3 document: ["heat", "action", "crime", "thriller", "pacino"]
        
        Vocabulary: ["action", "adventure", "animation", "board", "children", "comedy", "crime", 
                     "fantasy", "game", "heat", "jumanji", "pacino", "pixar", "story", "thriller", "toy"]
        
        TF calculation example for Movie 1:
        tokens = ["toy", "story", "animation", "adventure", "comedy", "pixar", "children"]
        token_counts = {"toy": 1, "story": 1, "animation": 1, "adventure": 1, "comedy": 1, "pixar": 1, "children": 1}
        doc_length = 7
        
        tf["toy"] = 1/7 = 0.143
        tf["story"] = 1/7 = 0.143
        tf["animation"] = 1/7 = 0.143
        (and so on)
        
        IDF calculation example:
        Total documents = 3
        "action" appears in 2 documents (Movie 2, Movie 3)
        idf["action"] = log(3/(1+2)) = log(1) = 0
        
        "animation" appears in 1 document (Movie 1)
        idf["animation"] = log(3/(1+1)) = log(1.5) = 0.176
        
        TF-IDF calculation for Movie 1:
        tfidf["toy"] = 0.143 * log(3/1) = 0.143 * 1.099 = 0.157
        tfidf["animation"] = 0.143 * 0.176 = 0.025
        
        Similarity calculation example:
        Similarity between Movie 1 and Movie 2:
        Common terms: "adventure"
        Movie 1 vector (partial): {"adventure": 0.143 * 0.176 = 0.025}
        Movie 2 vector (partial): {"adventure": 0.167 * 0.176 = 0.029}
        
        Dot product = 0.025 * 0.029 = 0.000725
        Magnitude1 = sqrt(0.157² + 0.025² + ...) = 0.3
        Magnitude2 = sqrt(0.029² + ...) = 0.35
        
        Similarity = 0.000725 / (0.3 * 0.35) = 0.000725 / 0.105 = 0.0069
        
        Final similarity matrix:
        sim[1, 1] = 1.0
        sim[1, 2] = 0.0069
        sim[1, 3] = 0.0
        sim[2, 1] = 0.0069
        sim[2, 2] = 1.0
        sim[2, 3] = 0.0288
        sim[3, 1] = 0.0
        sim[3, 2] = 0.0288
        sim[3, 3] = 1.0
        """
        print("Loading movies and tags data...")
        
        # Load movies data
        movies_df = pd.read_csv(movies_file)                         # -> Load movie information (title, genres)
        
        # Load tags data
        tags_df = pd.read_csv(tags_file)                             # -> Load user-generated tags for movies
        
        # Process ratings data
        all_movie_ids = set()                                        # -> Create a set to track all unique movie IDs
        
        for user_id, movie_id, rating in ratings_data:               # -> Process each rating in the input data
            # Initialize user dictionary if needed
            if user_id not in self.user_ratings:                     # -> Create dictionary for user if it doesn't exist yet
                self.user_ratings[user_id] = {}
            
            # Store rating
            self.user_ratings[user_id][movie_id] = rating            # -> Store the user's rating for the movie
            all_movie_ids.add(movie_id)                              # -> Add the movie ID to the set of all movie IDs
        
        # Create movie ID mappings
        self.movie_ids = sorted(list(all_movie_ids))                 # -> Create a sorted list of all movie IDs
        
        # Create dictionary mappings between IDs and indices
        self.movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}       # -> Map movie_id to array index
        self.movie_index_to_id = {idx: movie_id for movie_id, idx in self.movie_id_to_index.items()}  # -> Map array index to movie_id
        
        print(f"Building document corpus for {len(self.movie_ids)} movies...")
        
        # Create document for each movie (combining title, genres, and tags)
        for movie_id in self.movie_ids:                              # -> Process each movie to create its document
            # Get movie title and genres
            movie_info = movies_df[movies_df['movieId'] == movie_id] # -> Find the movie in the movies dataframe
            if len(movie_info) == 0:                                 # -> Skip if movie not found in the dataset
                continue
                
            title = movie_info.iloc[0]['title']                      # -> Extract the movie title
            genres = movie_info.iloc[0]['genres'].replace('|', ' ')  # -> Extract and process genres
            
            # Extract year from title if present
            year_match = re.search(r'\((\d{4})\)', title)            # -> Extract the year using regex
            year = ""
            if year_match:
                year = year_match.group(1)                           # -> Get the year if found
                # Remove year from title for processing
                title = re.sub(r'\(\d{4}\)', '', title).strip()      # -> Clean the title
            
            # Get tags for this movie
            movie_tags = tags_df[tags_df['movieId'] == movie_id]     # -> Find tags for this movie
            tags_text = " ".join(movie_tags['tag'].values) if len(movie_tags) > 0 else ""  # -> Join tags into a string
            
            # Combine all text
            document = f"{title} {genres} {tags_text} {year}"        # -> Create a single document combining all text
            
            # Preprocess the document
            tokens = self._preprocess_text(document)                 # -> Tokenize and clean the document
            
            # Store preprocessed document
            self.movie_documents[movie_id] = tokens                  # -> Save the preprocessed tokens for this movie
        
        # Build vocabulary from all tokens
        all_tokens = []                                              # -> Create a list to store all tokens
        for tokens in self.movie_documents.values():                 # -> Iterate through all movie documents
            all_tokens.extend(tokens)                                # -> Add tokens to the master list
        
        # Keep only tokens that appear in at least 2 documents to reduce noise
        token_counts = Counter(all_tokens)                           # -> Count how many times each token appears
        self.vocabulary = [token for token, count in token_counts.items() if count >= 2]  # -> Keep tokens appearing multiple times
        
        print(f"Vocabulary size: {len(self.vocabulary)}")
        
        # Build term-document matrix
        print("Computing TF-IDF vectors...")
        
        # Term frequency for each document
        tf = {}                                                      # -> Dictionary to store term frequencies
        for movie_id, tokens in self.movie_documents.items():        # -> Process each movie document
            tf[movie_id] = {}                                        # -> Initialize dictionary for this movie
            token_counts = Counter(tokens)                           # -> Count occurrences of each token
            doc_length = len(tokens)                                 # -> Get total length of document
            
            for token in token_counts:                               # -> Process each token in the document
                if token in self.vocabulary:                         # -> Only include tokens in our vocabulary
                    # TF = count(term in document) / total terms in document
                    tf[movie_id][token] = token_counts[token] / doc_length # -> Calculate normalized term frequency
        
        # Inverse document frequency
        idf = {}                                                     # -> Dictionary to store IDF values
        num_docs = len(self.movie_documents)                         # -> Total number of documents
        
        for term in self.vocabulary:                                 # -> Calculate IDF for each term in vocabulary
            # Count documents containing this term
            doc_count = sum(1 for movie_id in self.movie_documents if term in self.movie_documents[movie_id])
            # IDF = log(total docs / (docs containing term + 1))
            idf[term] = math.log(num_docs / (1 + doc_count))         # -> Calculate IDF with smoothing
        
        # Compute TF-IDF vectors
        for movie_id in self.movie_documents:                        # -> Create TF-IDF vector for each movie
            self.tfidf_vectors[movie_id] = {}                        # -> Initialize dictionary for this movie
            for term in self.vocabulary:                             # -> Process each term in vocabulary
                tf_value = tf[movie_id].get(term, 0)                 # -> Get term frequency (0 if not present)
                # TF-IDF = TF * IDF
                self.tfidf_vectors[movie_id][term] = tf_value * idf[term]  # -> Calculate TF-IDF score
        
        # Compute similarity matrix
        print("Computing TF-IDF similarity matrix...")
        n_items = len(self.movie_ids)                                # -> Number of movies
        self.similarities = np.zeros((n_items, n_items))             # -> Initialize similarity matrix with zeros
        
        # Track progress
        processed = 0                                                # -> Counter for processed pairs
        total = (n_items * (n_items - 1)) // 2                       # -> Total number of unique pairs
        
        for i in range(n_items):                                     # -> Iterate through each movie
            movie_id_i = self.movie_index_to_id[i]                   # -> Get movie ID for index i
            
            # Skip movies without TF-IDF vectors
            if movie_id_i not in self.tfidf_vectors:                 # -> Skip if no vector for this movie
                continue
                
            for j in range(i+1, n_items):                            # -> Compare with all other movies (only upper triangle)
                movie_id_j = self.movie_index_to_id[j]               # -> Get movie ID for index j
                
                # Skip if either movie has no TF-IDF vector
                if movie_id_j not in self.tfidf_vectors:             # -> Skip if no vector for this movie
                    continue
                
                # Compute cosine similarity
                similarity = self._compute_cosine_similarity(
                    self.tfidf_vectors[movie_id_i],                  # -> TF-IDF vector for movie i
                    self.tfidf_vectors[movie_id_j]                   # -> TF-IDF vector for movie j
                )
                
                # Store similarity (matrix is symmetric)
                self.similarities[i, j] = similarity                 # -> Store similarity in upper triangle
                self.similarities[j, i] = similarity                 # -> Mirror to lower triangle (symmetric matrix)
                
                processed += 1                                       # -> Increment processed counter
                if processed % 10000 == 0:                           # -> Show progress periodically
                    progress = (processed / total) * 100
                    print(f"Progress: {progress:.1f}% ({processed}/{total})")
        
        print("Similarity computation complete!")
        return self
    
    def _compute_cosine_similarity(self, vec1, vec2):
        """
        Compute cosine similarity between two sparse vectors represented as dictionaries
        
        Args:
            vec1: First vector as dictionary {term -> TF-IDF value}
            vec2: Second vector as dictionary {term -> TF-IDF value}
            
        Returns:
            float: Cosine similarity between the vectors
        """
        """
        Example calculation:
        vec1 = {"toy": 0.157, "story": 0.112, "animation": 0.025, "adventure": 0.015}
        vec2 = {"jumanji": 0.178, "adventure": 0.029, "fantasy": 0.134}
        
        Common terms: "adventure"
        
        Dot product = vec1["adventure"] * vec2["adventure"] = 0.015 * 0.029 = 0.000435
        
        Magnitude1 = sqrt(0.157² + 0.112² + 0.025² + 0.015²) = sqrt(0.03852) = 0.196
        Magnitude2 = sqrt(0.178² + 0.029² + 0.134²) = sqrt(0.04927) = 0.222
        
        Cosine similarity = 0.000435 / (0.196 * 0.222) = 0.000435 / 0.043512 = 0.01
        """
        # Find common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())                  # -> Find terms present in both vectors
        
        # Compute dot product for common terms
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms) # -> Dot product = sum(vec1[i] * vec2[i])
        
        # Compute magnitudes
        # Magnitude of vector = sqrt(sum of squared values)
        magnitude1 = math.sqrt(sum(value**2 for value in vec1.values()))    # -> Magnitude of first vector
        magnitude2 = math.sqrt(sum(value**2 for value in vec2.values()))    # -> Magnitude of second vector
        
        # Compute similarity
        if magnitude1 == 0 or magnitude2 == 0:                              # -> Avoid division by zero
            return 0
            
        return dot_product / (magnitude1 * magnitude2)                      # -> Cosine similarity formula: cos(θ) = A·B / (|A|·|B|)
    
    def predict(self, user_id, movie_id):
        """
        Predict rating for a user-movie pair
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating or None if prediction impossible
        """
        """
        Example prediction:
        
        User 1 has rated:
        - Movie 1: 5 stars
        - Movie 2: 3 stars
        
        We want to predict User 1's rating for Movie 3.
        
        Similarities:
        - sim(Movie 3, Movie 1) = 0.0
        - sim(Movie 3, Movie 2) = 0.0288
        
        Only Movie 2 has positive similarity with Movie 3, so neighbors = [(0.0288, 3)]
        
        Weighted average = (0.0288 * 3) / 0.0288 = 3.0
        
        Predicted rating for User 1, Movie 3 = 3.0
        """
        # Check if we can make prediction
        if user_id not in self.user_ratings or movie_id not in self.movie_id_to_index:  # -> Check for valid user and movie
            return None
            
        # Get index for target movie
        movie_idx = self.movie_id_to_index[movie_id]                        # -> Convert movie_id to matrix index
        
        # Get user's ratings
        user_rated_movies = self.user_ratings[user_id]                      # -> Get all movies rated by this user
        
        if not user_rated_movies:                                           # -> Check if user has rated any movies
            return None
            
        # Find similarities between target movie and all rated movies
        neighbors = []  # -> List to store (similarity, rating) pairs
        for rated_movie_id, rating in user_rated_movies.items():            # -> Process each movie rated by user
            # Skip if movie not in similarity matrix
            if rated_movie_id not in self.movie_id_to_index:
                continue
                
            rated_movie_idx = self.movie_id_to_index[rated_movie_id]        # -> Convert movie_id to matrix index
            similarity = self.similarities[movie_idx, rated_movie_idx]      # -> Look up similarity
            
            if similarity > 0:  # -> Only consider positive similarities
                neighbors.append((similarity, rating))
        
        if not neighbors:                                                   # -> Check if we found any similar movies
            return None
            
        # Get top-k neighbors
        k_neighbors = heapq.nlargest(min(self.k, len(neighbors)), neighbors, key=lambda x: x[0])  # -> Select k most similar
        
        # Calculate weighted average
        sim_total = sum(sim for sim, _ in k_neighbors)                      # -> Sum of all similarities
        if sim_total == 0:  # -> Check for zero division
            return None
            
        weighted_sum = sum(sim * rating for sim, rating in k_neighbors)     # -> Weighted sum of ratings
        predicted_rating = weighted_sum / sim_total                         # -> Weighted average
        
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
    
    """
    | Agorithms                       | RMSE   | MAE       |
    |---------------------------------|--------|-----------|
    | Random                          | 1.4385 |    1.1478 |
    | Content-Based Filtering         |--------|-----------|
    | TF-IDF                          | 0.9553 |    0.7421 | 
    """

    # Run evaluation
    print("Evaluating algorithms...")
    evaluator.Evaluate(False)
    
    # Generate sample recommendations
    evaluator.SampleTopNRecs(ml)