import sys
import os
# Get the absolute path to the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils.LoadMovieLensData import LoadMovieLensData

# For testing and evaluation
if __name__ == "__main__":    
    # Load data
    (ml, evaluationData, rankings) = LoadMovieLensData()

    print("Data loaded successfully.")
    print(f"ml: {ml}")
    
    # Print available methods and attributes
    print("\nAvailable methods and attributes:")
    for item in dir(ml):
        if not item.startswith('__'):
            print(f"  - {item}")
    
    # # Print some movie data using the correct methods
    # print("\nSample Movies:")
    # # The getMovieNames method doesn't exist, but we can use movieID_to_name dictionary
    # movies = ml.movieID_to_name
    # for i, (movie_id, movie_name) in enumerate(list(movies.items())[:10]):
    #     print(f"  {movie_id}: {movie_name}")
    
    # # Print popularity rankings
    # print("\nMost Popular Movies:")
    # popular_movies = sorted(rankings.items(), key=lambda x: x[1])[:10]
    # for movie_id, rank in popular_movies:
    #     if movie_id in movies:
    #         print(f"  Rank {rank}: {movies[movie_id]} (ID: {movie_id})")
    
    # # Print movie genres
    # print("\nMovie Genres:")
    # try:
    #     if hasattr(ml, 'getGenres'):
    #         genres = ml.getGenres()
    #         for i, (movie_id, genre_list) in enumerate(list(genres.items())[:5]):
    #             try:
    #                 movie_name = movies.get(movie_id, "Unknown")
    #                 # Convert integers to strings before joining
    #                 genre_strings = [str(g) for g in genre_list]
    #                 print(f"  {movie_name}: {', '.join(genre_strings)}")
    #             except Exception as e:
    #                 print(f"  Error processing movie {movie_id}: {e}")
    # except Exception as e:
    #     print(f"  Error getting genres: {e}")
    
    # # Print movie years if available
    # print("\nMovie Years:")
    # if hasattr(ml, 'getYears'):
    #     years = ml.getYears()
    #     for i, (movie_id, year) in enumerate(list(years.items())[:5]):
    #         movie_name = movies.get(movie_id, "Unknown")
    #         print(f"  {movie_name}: {year}")
    
    # # Print user ratings
    # print("\nSample User Ratings:")
    # if hasattr(ml, 'getUserRatings'):
    #     # Get ratings for a specific user (let's try user ID 1)
    #     user_id = 1
    #     user_ratings = ml.getUserRatings(user_id)
    #     print(f"  Ratings for User {user_id}:")
        
    #     # Fix: Check if user_ratings is a list or dict and handle accordingly
    #     if isinstance(user_ratings, dict):
    #         for i, (movie_id, rating) in enumerate(list(user_ratings.items())[:5]):
    #             movie_name = movies.get(movie_id, "Unknown")
    #             print(f"    {movie_name}: {rating}")
    #     elif isinstance(user_ratings, list):
    #         # If it's a list, assume it's a list of (movie_id, rating) tuples or similar structure
    #         for i, rating_item in enumerate(user_ratings[:5]):
    #             if isinstance(rating_item, tuple) and len(rating_item) >= 2:
    #                 movie_id, rating = rating_item[0], rating_item[1]
    #                 movie_name = movies.get(movie_id, "Unknown")
    #                 print(f"    {movie_name}: {rating}")
    #             else:
    #                 # If it's not a tuple, just print the whole item
    #                 print(f"    Rating {i+1}: {rating_item}")
    #     else:
    #         # Just print the first few elements however they come
    #         print(f"  User ratings format: {type(user_ratings)}")
    #         print(f"  First 5 ratings: {user_ratings[:5] if hasattr(user_ratings, '__getitem__') else user_ratings}")
        
    # # Print some mise en scene data if available
    # print("\nMise En Scene Data (Cinematography Features):")
    # if hasattr(ml, 'getMiseEnScene'):
    #     mise_data = ml.getMiseEnScene()
    #     # Just print the first few entries
    #     for i, (movie_id, features) in enumerate(list(mise_data.items())[:3]):
    #         movie_name = movies.get(movie_id, "Unknown")
    #         print(f"  {movie_name}:")
    #         print(f"    Features: {features}")

    # Print all attributes and their types
    print("\nAll MovieLens Data Structures:")
    for attr_name in dir(ml):
        if attr_name.startswith('__'):
            continue
            
        attr = getattr(ml, attr_name)
        if callable(attr):
            if attr_name not in ['getGenres', 'getMiseEnScene', 'getMovieID', 'getMovieName', 
                               'getPopularityRanks', 'getUserRatings', 'getYears', 
                               'loadMovieLensLatestSmall']:
                continue
                
            try:
                # Call certain getter methods without parameters
                if attr_name in ['getGenres', 'getMiseEnScene', 'getPopularityRanks', 'getYears']:
                    result = attr()
                    if isinstance(result, dict):
                        print(f"  {attr_name}: Dictionary with {len(result)} items")
                    else:
                        print(f"  {attr_name}: {type(result)}")
            except:
                # Skip methods that require parameters
                pass
        else:
            # Print non-callable attributes (variables)
            if isinstance(attr, dict):
                print(f"  {attr_name}: Dictionary with {len(attr)} items")
            elif isinstance(attr, str):
                print(f"  {attr_name}: {attr}")
            else:
                print(f"  {attr_name}: {type(attr)}")
                
    # Print the original CSV data structure
    print("\nOriginal MovieLens Dataset:")
    try:
        # Read the original CSV files to show column structure
        import pandas as pd
        
        # Read movies.csv
        movies_path = ml.moviesPath
        print(f"\nMovies file: {movies_path}")
        movies_df = pd.read_csv(movies_path)
        print(f"  Shape: {movies_df.shape}")
        print(f"  Columns: {', '.join(movies_df.columns)}")
        print("  Sample data:")
        print(movies_df.head(3).to_string())
        
        # Read ratings.csv
        ratings_path = ml.ratingsPath  
        print(f"\nRatings file: {ratings_path}")
        ratings_df = pd.read_csv(ratings_path)
        print(f"  Shape: {ratings_df.shape}")
        print(f"  Columns: {', '.join(ratings_df.columns)}")
        print("  Sample data:")
        print(ratings_df.head(3).to_string())
        
        # Read links.csv if it exists
        links_path = os.path.join(os.path.dirname(movies_path), 'links.csv')
        if os.path.exists(links_path):
            print(f"\nLinks file: {links_path}")
            links_df = pd.read_csv(links_path)
            print(f"  Shape: {links_df.shape}")
            print(f"  Columns: {', '.join(links_df.columns)}")
            print("  Sample data:")
            print(links_df.head(3).to_string())
            
        # Read tags.csv if it exists
        tags_path = os.path.join(os.path.dirname(movies_path), 'tags.csv')
        if os.path.exists(tags_path):
            print(f"\nTags file: {tags_path}")
            tags_df = pd.read_csv(tags_path)
            print(f"  Shape: {tags_df.shape}")
            print(f"  Columns: {', '.join(tags_df.columns)}")
            print("  Sample data:")
            print(tags_df.head(3).to_string())
        
    except Exception as e:
        print(f"  Error reading original CSV files: {e}")


    print("\n\n" + "="*50 + "\n")
    # Print details about the evaluationData object
    print("\nevaluationData Data Information:")
    print(f"Type: {type(evaluationData)}")
    
    # Print available methods and attributes in evaluationData
    print("\nEvaluation Data methods and attributes:")
    for item in dir(evaluationData):
        if not item.startswith('__'):
            print(f"  - {item}")
    
    # For Surprise dataset, display information differently
    print("\nSurprise Dataset Details:")
    
    # Display raw ratings information
    if hasattr(evaluationData, 'raw_ratings'):
        raw_ratings = evaluationData.raw_ratings
        print(f"  Number of raw ratings: {len(raw_ratings)}")
        print("  Sample raw ratings (first 5):")
        for i, (user, item, rating, timestamp) in enumerate(raw_ratings[:5]):
            movie_name = ml.movieID_to_name.get(int(item), "Unknown")
            print(f"    User {user} rated {movie_name} as {rating} at {timestamp}")
    
    # Build and display trainset information
    if hasattr(evaluationData, 'build_full_trainset'):
        print("\nBuilding full trainset...")
        try:
            trainset = evaluationData.build_full_trainset()
            print(f"  Number of users: {trainset.n_users}")
            print(f"  Number of items: {trainset.n_items}")
            print(f"  Number of ratings: {trainset.n_ratings}")
            print(f"  Rating scale: {trainset.rating_scale}")
            
            # Display sample ratings from trainset
            print("\n  Sample trainset ratings:")
            for (u_id, i_id, rating) in list(trainset.all_ratings())[:5]:
                # Convert internal IDs to raw IDs
                raw_uid = trainset.to_raw_uid(u_id)
                raw_iid = trainset.to_raw_iid(i_id)
                movie_name = ml.movieID_to_name.get(int(raw_iid), "Unknown")
                print(f"    User {raw_uid} rated {movie_name} as {rating}")
            
            # Show user and item information
            print("\n  User information:")
            u_ids = list(trainset.all_users())[:3]
            for u_id in u_ids:
                raw_uid = trainset.to_raw_uid(u_id)
                user_items = trainset.ur[u_id]
                print(f"    User {raw_uid} has rated {len(user_items)} movies")
            
            print("\n  Item information:")
            i_ids = list(trainset.all_items())[:3]
            for i_id in i_ids:
                raw_iid = trainset.to_raw_iid(i_id)
                movie_name = ml.movieID_to_name.get(int(raw_iid), "Unknown")
                item_users = trainset.ir[i_id]
                print(f"    Movie {movie_name} has been rated by {len(item_users)} users")
        
        except Exception as e:
            print(f"  Error building trainset: {e}")
    
    # Show reader information
    if hasattr(evaluationData, 'reader'):
        reader = evaluationData.reader
        print("\nReader information:")
        print(f"  Rating scale: {reader.rating_scale}")
        
        # Check for attributes before trying to access them
        if hasattr(reader, 'line_format'):
            print(f"  Line format: {reader.line_format}")
        
        if hasattr(reader, 'skip_lines'):
            print(f"  Skip lines: {reader.skip_lines}")
            
        if hasattr(reader, 'sep'):
            print(f"  Separator: '{reader.sep}'")
            
        # Print all reader attributes that don't start with underscore
        print("  All reader attributes:")
        for attr_name in dir(reader):
            if not attr_name.startswith('__'):
                try:
                    attr_value = getattr(reader, attr_name)
                    # Only print if it's not a method
                    if not callable(attr_value):
                        print(f"    - {attr_name}: {attr_value}")
                except:
                    print(f"    - {attr_name}: <Unable to access>")


    print("\n\n" + "="*50 + "\n")
    # Print details about the evaluationData object
    print("\nrankings Data Information:")
    print(f"Type: {type(rankings)}")
    
    # Print available methods and attributes in evaluationData
    print("\nRankings Data methods and attributes:")
    for item in dir(evaluationData):
        if not item.startswith('__'):
            print(f"  - {item}")
    
    # Fix: These were evaluationData methods, not rankings methods
    print("\nRankings Data structure:")
    
    # Check if rankings is a dictionary-like object
    if hasattr(rankings, 'items'):
        # Get the number of items
        num_items = len(rankings)
        print(f"  Number of items: {num_items}")
        
        # Check the type of keys and values
        sample_items = list(rankings.items())[:5]
        if sample_items:
            first_key, first_value = sample_items[0]
            print(f"  Key type: {type(first_key).__name__}")
            print(f"  Value type: {type(first_value).__name__}")
            
        # Print some sample items
        print("\n  Sample rankings (first 10 items):")
        for movie_id, rank in sorted(list(rankings.items()), key=lambda x: x[1])[:10]:
            if hasattr(ml, 'movieID_to_name') and movie_id in ml.movieID_to_name:
                movie_name = ml.movieID_to_name[movie_id]
                print(f"    Movie ID: {movie_id}, Rank: {rank}, Title: {movie_name}")
            else:
                print(f"    Movie ID: {movie_id}, Rank: {rank}")
                
        # Print some statistics
        ranks = list(rankings.values())
        if ranks:
            print(f"\n  Min rank: {min(ranks)}")
            print(f"  Max rank: {max(ranks)}")
            
        # Explain what the rankings represent
        print("\n  Note: These rankings represent the popularity of movies based on how many users have rated them.")
        print("        Lower rank numbers (1, 2, 3, etc.) indicate more popular movies.")
    else:
        print("  rankings is not a dictionary-like object")
        print(f"  Object representation: {rankings}")
        
    # Since rankings doesn't have columns, show the related CSV files that would have columns
    print("\nRelated datasets with columns:")
    print("  1. movies.csv: movieId, title, genres")
    print("  2. ratings.csv: userId, movieId, rating, timestamp")
    print("  3. links.csv: movieId, imdbId, tmdbId")
    print("  4. tags.csv: userId, movieId, tag, timestamp")