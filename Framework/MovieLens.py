import os
import csv
import sys
import re

from surprise import Dataset
from surprise import Reader

from collections import defaultdict
import numpy as np

class MovieLens:
    """
    Helper class to load and process the MovieLens dataset ("latest small").

    Attributes:
        movieID_to_name (dict[int, str]): Mapping from movie ID to movie name.
        name_to_movieID (dict[str, int]): Mapping from movie name to movie ID.
        ratingsPath (str):                File path to the ratings CSV.
        moviesPath (str):                 File path to the movies CSV.
    """

    movieID_to_name = {}
    name_to_movieID = {}
    ratingsPath = '../ml-latest-small/ratings.csv'
    moviesPath = '../ml-latest-small/movies.csv'
    
    def loadMovieLensLatestSmall(self):
        """
        Load the MovieLens "latest small" dataset for use with Surprise.

        - Reads ratings from a CSV using Surprise's Reader class.
        - Builds mappings between movie IDs and names from the movies CSV.

        Returns:
            Dataset: A Surprise Dataset object containing the ratings.
        """
        # Ensure relative paths work by changing to script directory
        os.chdir(os.path.dirname(sys.argv[0]))

        # Reset mappings
        ratingsDataset = 0
        self.movieID_to_name = {}
        self.name_to_movieID = {}

        # Check if the files exist
        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

        # Check if the ratings file exists
        ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)

        # Parse the movies file to build lookup dictionaries
        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
                movieReader = csv.reader(csvfile)
                next(movieReader)  #Skip header line
                for row in movieReader:
                    movieID = int(row[0])
                    movieName = row[1]
                    self.movieID_to_name[movieID] = movieName
                    self.name_to_movieID[movieName] = movieID

        return ratingsDataset

    def getUserRatings(self, user):
        """
        Retrieve all (movieID, rating) pairs for a specific user from the ratings CSV.

        Args:
            user (int): The user ID to filter ratings by.

        Returns:
            list of (movieID, rating): Ratings given by the specified user, in order.
        """
        userRatings = []
        hitUser = False
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader) # Skip header
            for row in ratingReader:
                userID = int(row[0])
                if (user == userID):
                    # Collect ratings until user changes
                    movieID = int(row[1])
                    rating = float(row[2])
                    userRatings.append((movieID, rating))
                    hitUser = True
                if (hitUser and (user != userID)):
                    # Once we have passed this user's block, stop reading
                    break
        return userRatings

    def getPopularityRanks(self):
        """
        Compute popularity rank for each movie based on number of ratings.

        Reads the ratings CSV, counts total ratings per movie,
        then ranks movies from most-rated (1) to least-rated.

        Returns:
            dict: Mapping movieID -> rank (1 is most popular).
        """
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        # Count how many times each movie appears in ratings
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                movieID = int(row[1])
                ratings[movieID] += 1
        # Sort movies by rating count (descending) to assign ranks
        rank = 1
        for movieID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[movieID] = rank
            rank += 1
        return rankings
    
    def getGenres(self):
        """
        Extract genre information for each movie as a binary vector.

        - Parses the movies CSV, building a unique integer ID for each genre.
        - Converts each movie's list of genre IDs into a bitfield list.

        Returns:
            dict: Mapping movieID -> binary genre vector.
        """
        genres = defaultdict(list)
        genreIDs = {}
        maxGenreID = 0
        # First pass: assign an integer ID to each genre
        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)  #Skip header line
            for row in movieReader:
                movieID = int(row[0])
                genreList = row[2].split('|')
                genreIDList = []
                for genre in genreList:
                    if genre in genreIDs:
                        genreID = genreIDs[genre]
                    else:
                        genreID = maxGenreID
                        genreIDs[genre] = genreID
                        maxGenreID += 1
                    genreIDList.append(genreID)
                genres[movieID] = genreIDList
        # Second pass: convert genre ID lists into fixed-length bitfields
        for (movieID, genreIDList) in genres.items():
            bitfield = [0] * maxGenreID
            for genreID in genreIDList:
                bitfield[genreID] = 1
            genres[movieID] = bitfield            
        
        return genres
    
    def getYears(self):
        """
        Extract release year for each movie from its title.

        - Uses regex to find a four-digit year in parentheses at the end of the title.
        - If found, maps movieID to that year.

        Returns:
            dict: Mapping movieID -> release year.
        """
        # Regex matches '(YYYY)' at end of title
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)
        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)
            for row in movieReader:
                movieID = int(row[0])
                title = row[1]
                m = p.search(title)
                year = m.group(1)
                if year:
                    years[movieID] = int(year)
        return years
    
    def getMiseEnScene(self):
        """
        Load precomputed visual "mise-en-scene" features for movies.

        Reads a CSV with columns:
            movieID, 
            avgShotLength, 
            meanColorVariance,
            stddevColorVariance, 
            meanMotion, 
            stddevMotion,
            meanLightingKey, 
            numShots

        Returns:
            dict: Mapping movieID -> list of feature values.
        """
        mes = defaultdict(list)
        with open("LLVisualFeatures13K_Log.csv", newline='') as csvfile:
            mesReader = csv.reader(csvfile)
            next(mesReader)
            for row in mesReader:
                movieID = int(row[0])
                avgShotLength = float(row[1])
                meanColorVariance = float(row[2])
                stddevColorVariance = float(row[3])
                meanMotion = float(row[4])
                stddevMotion = float(row[5])
                meanLightingKey = float(row[6])
                numShots = float(row[7])
                mes[movieID] = [avgShotLength, meanColorVariance, stddevColorVariance,
                   meanMotion, stddevMotion, meanLightingKey, numShots]
        return mes
    
    def getMovieName(self, movieID):
        """
        Retrieve a movie's name given its ID.

        Args:
            movieID (int): The ID of the movie.

        Returns:
            str: Movie name, or empty string if not found.
        """
        if movieID in self.movieID_to_name:
            return self.movieID_to_name[movieID]
        else:
            return ""
        
    def getMovieID(self, movieName):
        """
        Retrieve a movie's ID given its name.

        Args:
            movieName (str): The name of the movie.

        Returns:
            int: Movie ID, or 0 if not found.
        """
        if movieName in self.name_to_movieID:
            return self.name_to_movieID[movieName]
        else:
            return 0