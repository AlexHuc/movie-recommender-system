import sys
import os
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import glob

# Set up a Dark Theme
plt.style.use(['dark_background', 'ggplot'])

# Color Palette
neon_palette = [
    '#FF9500',                     # Neon orange
    '#00F2C3',                     # Bright teal
    '#FF3B30',                     # Vibrant red
    '#5AC8FA',                     # Electric blue
    '#AF52DE',                     # Bright purple
    '#FFCC00',                     # Golden yellow
    '#34C759',                     # Lime green
    '#007AFF',                     # Azure blue
    '#FF6482',                     # Salmon pink
    '#C4E17F'                      # Chartreuse
]

# Style
plt.rcParams.update({
    # Font styling
    'font.family': 'Avenir, Helvetica, Arial',
    'font.weight': 'medium',         # Medium weight for readability
    'font.size': 11,                 # Base font size for consistency
    'axes.titleweight': 'bold',      # Bold titles for axes
    'axes.titlesize': 16,            # Larger title for axes
    'axes.labelweight': 'bold',      # Bold labels for emphasis
    'axes.labelsize': 13,            # Consistent label size
    'xtick.labelsize': 11,           # Consistent tick label size
    'ytick.labelsize': 11,           # Consistent tick label size
    'legend.fontsize': 11,           # Consistent legend font size
    'figure.titlesize': 18,          # Larger title for figures
    
    # Figure properties
    'figure.figsize': (12, 7),       # Default figure size
    'figure.dpi': 120,               # Higher DPI for better quality
    'figure.facecolor': '#1A1A1A', # Nearly black background

    # Axes styling
    'axes.facecolor': '#212121',   # Dark gray with slight warmth
    'axes.edgecolor': '#444444',   # Medium gray edges
    'axes.linewidth': 1.5,           # Thicker axis lines
    'axes.grid': True,               # Enable grid
    'axes.axisbelow': True,          # Grid behind data
    'axes.labelpad': 10,             # More padding for labels
    'axes.spines.top': False,        # Remove top spine
    'axes.spines.right': False,      # Remove right spine
    
    # Grid styling
    'grid.color': '#404040',       # Darker grid lines
    'grid.linestyle': '--',          # Dashed grid lines
    'grid.linewidth': 0.8,           # Thinner grid lines
    'grid.alpha': 0.5,               # Slightly transparent grid lines
    
    # Tick styling
    'xtick.color': '#BBBBBB',      # Light gray ticks
    'ytick.color': '#BBBBBB',      # Light gray ticks
    'xtick.major.size': 7,           # Larger ticks
    'ytick.major.size': 7,           # Larger ticks
    'xtick.major.width': 1.5,        # Thicker major ticks
    'ytick.major.width': 1.5,        # Thicker major ticks
    
    # Text styling
    'text.color': '#EEEEEE',       # Off-white text
    'axes.labelcolor': '#EEEEEE',  # Off-white axes labels
    
    # Color cycles
    'axes.prop_cycle': cycler(color=neon_palette),
    
    # Legend styling
    'legend.fancybox': True,         # Rounded corners for legend box
    'legend.framealpha': 0.8,        # Slightly transparent legend box
    'legend.edgecolor': '#444444', # Medium gray edge for legend box
    'legend.borderpad': 0.8,         # Padding inside legend box
    
    # Saving figures
    'savefig.dpi': 150,              # Higher DPI for saved figures
    'savefig.bbox': 'tight',         # Tight bounding box for saved figures
    'savefig.facecolor': '#1A1A1A',# Nearly black background for saved figures
})

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import LoadMovieLensData
from utils.LoadMovieLensData import LoadMovieLensData

import importlib

try:
    # Collaborative Filtering Algorithms
    item_knn_module = importlib.import_module("1_ItemBasedCollaborativeFiltering.ItemKNN")
    AdaptedItemKNN = getattr(item_knn_module, "AdaptedItemKNN")
    PureItemKNN = getattr(item_knn_module, "PureItemKNN")

    user_knn_module = importlib.import_module("1_UserBasedCollaborativeFiltering.UserKNN")
    AdaptedUserKNN = getattr(user_knn_module, "AdaptedUserKNN")
    PureUserKNN = getattr(user_knn_module, "PureUserKNN")

    # Matrix Factorization Algorithms
    als_module = importlib.import_module("2_MatrixFactorization.ALS")
    AdaptedALS = getattr(als_module, "AdaptedALS")
    PureALS = getattr(als_module, "PureALS")
    
    nmf_module = importlib.import_module("2_MatrixFactorization.NMF")
    AdaptedNMF = getattr(nmf_module, "AdaptedNMF")
    PureNMF = getattr(nmf_module, "PureNMF")
    
    pmf_module = importlib.import_module("2_MatrixFactorization.PMF")
    AdaptedPMF = getattr(pmf_module, "AdaptedPMF")
    PurePMF = getattr(pmf_module, "PurePMF")
    
    svd_module = importlib.import_module("2_MatrixFactorization.SVD")
    AdaptedSVD = getattr(svd_module, "AdaptedSVD")
    PureSVD = getattr(svd_module, "PureSVD")
    
    svdpp_module = importlib.import_module("2_MatrixFactorization.SVDpp")
    AdaptedSVDpp = getattr(svdpp_module, "AdaptedSVDpp")
    PureSVDpp = getattr(svdpp_module, "PureSVDpp")

    # Content-Based Filtering Algorithms
    svdpp_module = importlib.import_module("3_ContentBased.TFIDF")
    AdaptedTFIDF = getattr(svdpp_module, "AdaptedTFIDF")
    PureTFIDF = getattr(svdpp_module, "PureTFIDF")

    print(f"Successfully imported all the ML algorithms")
except (ImportError, AttributeError) as e:
    print(f"Import failed: {e}")

# Page setup
st.set_page_config(page_title="MovieLens Explorer", layout="wide")
st.title("🎬 MovieLens Data Explorer")
st.markdown("This application shows various aspects of the MovieLens dataset.")

# Load data
try:
    with st.spinner('Loading MovieLens data... This might take a moment.'):
        (ml, evaluationData, rankings) = LoadMovieLensData()
    
    st.success("Data loaded successfully!")
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Show tabs for different data structures
tabs = st.tabs(["🎬 Movies", "⭐ Ratings", "🏆 Rankings", "🎭 Genres", "📅 Years", "🔄 Additional Details", "🤖 Recommendation Systems"])

# Tab 1: Movies
with tabs[0]:
    st.header("Movies")
    
    # Get movie dictionary
    movies_dict = ml.movieID_to_name
    
    # Convert to dataframe
    movies_df = pd.DataFrame(list(movies_dict.items()), columns=['movieId', 'title'])
    
    # Add statistics
    st.write(f"Total movies: {len(movies_dict)}")
    
    # Search box
    movie_search = st.text_input("Search for movies")
    
    if movie_search:
        filtered_movies = movies_df[movies_df['title'].str.contains(movie_search, case=False)]
        st.dataframe(filtered_movies, use_container_width=True)
    else:
        # Show all movies with pagination
        st.subheader("All Movies")
        
        # Add sorting options
        sort_options = ["Title (A-Z)", "Title (Z-A)", "ID (Ascending)", "ID (Descending)"]
        sort_choice = st.radio("Sort by:", sort_options, horizontal=True)
        
        # Sort the dataframe based on selection
        if sort_choice == "Title (A-Z)":
            sorted_df = movies_df.sort_values('title')
        elif sort_choice == "Title (Z-A)":
            sorted_df = movies_df.sort_values('title', ascending=False)
        elif sort_choice == "ID (Ascending)":
            sorted_df = movies_df.sort_values('movieId')
        else:  # ID (Descending)
            sorted_df = movies_df.sort_values('movieId', ascending=False)
        
        # Display the full dataframe with pagination
        st.dataframe(sorted_df, use_container_width=True)
        
        # Show count at the bottom
        st.write(f"Displaying all {len(movies_df)} movies")
    
    # Plot release years (extracting from titles)
    st.subheader("Movie Release Years")
    
    # Extract years from titles using regex
    import re
    def extract_year(title):
        match = re.search(r'\((\d{4})\)', title)
        if match:
            return int(match.group(1))
        return None
    
    movies_df['year'] = movies_df['title'].apply(extract_year)
    
    # Create a histogram of movie years
    year_counts = movies_df['year'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(year_counts.index, year_counts.values)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Movies')
    ax.set_title('Movies by Release Year')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# Tab 2: Ratings
with tabs[1]:
    st.header("User Ratings")
    
    # Get a sample user
    user_ids = list(range(1, 672))  # MovieLens typically has users 1-671
    selected_user = st.selectbox("Select User ID", user_ids)
    
    # Get user ratings
    if hasattr(ml, 'getUserRatings'):
        user_ratings = ml.getUserRatings(selected_user)
        
        if isinstance(user_ratings, dict):
            # Convert to dataframe
            user_df = pd.DataFrame(list(user_ratings.items()), columns=['movieId', 'rating'])
        elif isinstance(user_ratings, list):
            # If it's a list of tuples
            if user_ratings and isinstance(user_ratings[0], tuple):
                user_df = pd.DataFrame(user_ratings, columns=['movieId', 'rating'])
            else:
                st.error("Unexpected format for user ratings")
                user_df = pd.DataFrame()
        else:
            st.error(f"Unexpected type for user ratings: {type(user_ratings)}")
            user_df = pd.DataFrame()
    
        if not user_df.empty:
            # Show the number of movies watched by this user
            movie_count = len(user_df)
            st.write(f"User {selected_user} has rated {movie_count} movies")
            
            # Join with movie titles
            user_df['title'] = user_df['movieId'].map(ml.movieID_to_name)
            
            # Sort by rating
            user_df = user_df.sort_values('rating', ascending=False)
            
            # Display dataframe
            st.dataframe(user_df[['title', 'rating']], use_container_width=True)
            
            # Rating distribution
            st.subheader("Rating Distribution")
            rating_counts = user_df['rating'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(rating_counts.index.astype(str), rating_counts.values)
            ax.set_xlabel('Rating')
            ax.set_ylabel('Number of Movies')
            ax.set_title(f'Rating Distribution for User {selected_user}')
            st.pyplot(fig)

            # Add genre analysis for the user
            st.subheader(f"Genre Preferences for User {selected_user}")

            # Load movies data to get genre information
            movies_path = os.path.join(parent_dir, 'ml-latest-small', 'movies.csv')
            if os.path.exists(movies_path):
                # Load movies with genres
                movies_with_genres = pd.read_csv(movies_path)
                
                # Merge with user's ratings
                user_movies_with_genres = pd.merge(
                    user_df, 
                    movies_with_genres, 
                    on='movieId', 
                    how='left',
                    suffixes=('', '_full')
                )
                
                # Extract all genres that the user has rated
                user_genres = []
                for genres_str in user_movies_with_genres['genres']:
                    if isinstance(genres_str, str):  # Check if it's a valid string
                        user_genres.extend(genres_str.split('|'))
                
                if user_genres:
                    # Count each genre
                    genre_counts = pd.Series(user_genres).value_counts()
                    
                    # Plot the genre distribution
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Sort by count for better visualization
                    genre_counts = genre_counts.sort_values(ascending=False)
                    
                    # Use horizontal bar chart for better readability with many genres
                    ax.barh(genre_counts.index, genre_counts.values, color='lightcoral')
                    ax.set_xlabel('Number of Movies')
                    ax.set_ylabel('Genre')
                    ax.set_title(f'Genres Watched by User {selected_user}')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show highest rated genres
                    st.subheader(f"Average Rating by Genre for User {selected_user}")
                    
                    # Calculate average rating per genre
                    genre_ratings = {}
                    
                    for _, row in user_movies_with_genres.iterrows():
                        if isinstance(row['genres'], str):
                            movie_genres = row['genres'].split('|')
                            for genre in movie_genres:
                                if genre not in genre_ratings:
                                    genre_ratings[genre] = []
                                genre_ratings[genre].append(row['rating'])
                    
                    # Calculate average for each genre
                    genre_avg_ratings = {
                        genre: sum(ratings) / len(ratings) 
                        for genre, ratings in genre_ratings.items() 
                        if len(ratings) > 0  # Avoid division by zero
                    }
                    
                    # Convert to DataFrame for easier plotting
                    genre_avg_df = pd.DataFrame({
                        'Genre': list(genre_avg_ratings.keys()),
                        'Average Rating': list(genre_avg_ratings.values()),
                        'Movie Count': [len(genre_ratings[genre]) for genre in genre_avg_ratings.keys()]
                    })
                    
                    # Sort by average rating
                    genre_avg_df = genre_avg_df.sort_values('Average Rating', ascending=False)
                    
                    # Only show genres with at least 2 movies
                    genre_avg_df = genre_avg_df[genre_avg_df['Movie Count'] >= 2]
                    
                    if not genre_avg_df.empty:
                        # Plot average ratings
                        fig, ax = plt.subplots(figsize=(12, 8))
                        bars = ax.barh(genre_avg_df['Genre'], genre_avg_df['Average Rating'], color='lightgreen')
                        
                        # Add the count as text annotations
                        for i, bar in enumerate(bars):
                            count = genre_avg_df.iloc[i]['Movie Count']
                            ax.text(
                                bar.get_width() + 0.1, 
                                bar.get_y() + bar.get_height()/2,
                                f"({count} movies)",
                                va='center'
                            )
                        
                        ax.set_xlabel('Average Rating')
                        ax.set_ylabel('Genre')
                        ax.set_title(f'Average Rating by Genre for User {selected_user}')
                        ax.set_xlim(0, 5.5)  # Ratings are typically 0-5
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.write("Not enough rated movies per genre to calculate averages.")
                else:
                    st.write("No genre information available for this user's movies.")
            else:
                st.write("Could not load genre information.")
            
            # Top rated movies
            st.subheader("Top Rated Movies")
            top_movies = user_df.head(10)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.barh(top_movies['title'].str[:30], top_movies['rating'], color='skyblue')
            ax.set_xlabel('Rating')
            ax.set_ylabel('Movie')
            ax.set_title(f'Top Rated Movies for User {selected_user}')
            ax.invert_yaxis()  # To show highest rated at top
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.write("No ratings found for this user.")
    else:
        st.error("getUserRatings method not available in the MovieLens object")

# Tab 3: Rankings
with tabs[2]:
    st.header("Movie Popularity Rankings")
    
    # Get rankings
    if rankings:
        # Convert to dataframe
        rank_df = pd.DataFrame(list(rankings.items()), columns=['movieId', 'rank'])
        
        # Add titles
        rank_df['title'] = rank_df['movieId'].map(ml.movieID_to_name)
        
        # Sort by rank
        rank_df = rank_df.sort_values('rank')
        
        # Display statistics
        st.write(f"Number of ranked movies: {len(rank_df)}")
        st.write(f"Rank range: 1 (most popular) to {rank_df['rank'].max()} (least popular)")
        
        # Display top ranked movies
        st.subheader("Top 20 Most Popular Movies")
        st.dataframe(rank_df[['rank', 'title']].head(20), use_container_width=True)
        
        # Plot top 10
        st.subheader("Top 10 Most Popular Movies")
        top_10 = rank_df.head(10)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(top_10['title'].str[:30], top_10['rank'].max() - top_10['rank'] + 1, color='coral')
        ax.set_xlabel('Popularity Score (inverted rank)')
        ax.set_ylabel('Movie')
        ax.set_title('Top 10 Most Popular Movies')
        ax.invert_yaxis()  # To show highest ranked at top
        plt.tight_layout()
        st.pyplot(fig)
        
        # Search for a movie's rank
        st.subheader("Search for Movie Rank")
        movie_search = st.text_input("Enter movie title to find its popularity rank")
        
        if movie_search:
            filtered_ranks = rank_df[rank_df['title'].str.contains(movie_search, case=False)]
            if not filtered_ranks.empty:
                st.dataframe(filtered_ranks[['rank', 'title']], use_container_width=True)
            else:
                st.write("No matching movies found.")
    else:
        st.error("Rankings data not available")

# Tab 4: Genres
with tabs[3]:
    st.header("Movie Genres")
    
    # Create a mapping of genre names based on MovieLens documentation
    genre_names = [
        "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
        "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western", "(no genres listed)"
    ]
    
    # Load the movies CSV directly to get actual genre strings
    movies_path = os.path.join(parent_dir, 'ml-latest-small', 'movies.csv')
    if os.path.exists(movies_path):
        movies_df = pd.read_csv(movies_path)
        
        # Display statistics
        st.write(f"Total movies: {len(movies_df)}")
        
        # Get genre distribution
        all_genres = []
        for genre_list in movies_df['genres']:
            all_genres.extend(genre_list.split('|'))
        
        genre_counts = pd.Series(all_genres).value_counts()
        
        # Plot genre distribution
        st.subheader("Genre Distribution")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(genre_counts.index, genre_counts.values, color='skyblue')
        ax.set_xlabel('Number of Movies')
        ax.set_ylabel('Genre')
        ax.set_title('Movies per Genre')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Search by genre
        st.subheader("Search Movies by Genre")
        selected_genre = st.selectbox("Select a Genre", sorted(genre_counts.index))
        
        if selected_genre:
            genre_movies = movies_df[movies_df['genres'].str.contains(selected_genre)]
            st.write(f"Found {len(genre_movies)} movies in the {selected_genre} genre")
            st.dataframe(genre_movies[['movieId', 'title', 'genres']], use_container_width=True)
        
        # Multi-genre analysis
        st.subheader("Multi-genre Analysis")
        
        # Count movies by number of genres
        movies_df['genre_count'] = movies_df['genres'].apply(lambda x: len(x.split('|')))
        genre_count_dist = movies_df['genre_count'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(genre_count_dist.index, genre_count_dist.values)
        ax.set_xlabel('Number of Genres')
        ax.set_ylabel('Number of Movies')
        ax.set_title('Movies by Number of Genres')
        plt.xticks(genre_count_dist.index)
        plt.tight_layout()
        st.pyplot(fig)

        # Show genre combinations by size
        st.subheader("Common Genre Combinations")

        # Allow user to select the number of genres to combine
        max_genres = min(8, movies_df['genre_count'].max())  # Up to 6 or max available
        combo_sizes = list(range(2, max_genres + 1))
        selected_combo_size = st.select_slider("Number of genres in combination:", 
                                            options=combo_sizes, value=2)

        # Collect combinations of the selected size
        st.write(f"Showing top combinations of {selected_combo_size} genres:")

        # Dictionary to store combinations for each size
        all_combos = {size: [] for size in range(2, max_genres + 1)}

        # Get combinations of different sizes
        for genre_list in movies_df['genres']:
            genres = genre_list.split('|')
            genres = [g for g in genres if g != "(no genres listed)"]  # Remove no genres
            
            # Get combinations of different sizes
            if len(genres) >= 2:
                # For pairs (2 genres)
                from itertools import combinations
                for combo_size in range(2, min(len(genres) + 1, max_genres + 1)):
                    for combo in combinations(genres, combo_size):
                        # Sort to avoid duplicates
                        sorted_combo = tuple(sorted(combo))
                        all_combos[combo_size].append(sorted_combo)

        # Process the selected combination size
        if selected_combo_size in all_combos:
            combo_series = pd.Series(all_combos[selected_combo_size])
            combo_counts = combo_series.value_counts()
            
            # Limit to top 20 for display
            top_n = min(20, len(combo_counts))
            top_combos = combo_counts.head(top_n)
            
            # Format for display
            combo_df = pd.DataFrame({
                'Genres': [' + '.join(combo) for combo in top_combos.index],
                'Count': top_combos.values
            })
            
            # Show the dataframe
            st.write(f"Top {top_n} combinations of {selected_combo_size} genres:")
            st.dataframe(combo_df, use_container_width=True)
            
            # Draw a chart for the top 10
            fig, ax = plt.subplots(figsize=(12, 8))
            display_count = min(10, len(combo_df))  # Limit to 10 for better chart readability
            chart_data = combo_df.head(display_count)
            
            ax.barh(chart_data['Genres'], chart_data['Count'], color='lightgreen')
            ax.set_xlabel('Number of Movies')
            ax.set_ylabel('Genre Combination')
            ax.set_title(f'Most Common Combinations of {selected_combo_size} Genres')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show example movies with this combination
            if not combo_df.empty:
                st.subheader(f"Example Movies with '{combo_df['Genres'].iloc[0]}'")
                # Create a pattern to search for all genres in the combination
                top_combo = top_combos.index[0]
                
                # Find movies containing all genres in the combination
                example_movies = movies_df.copy()
                for genre in top_combo:
                    example_movies = example_movies[example_movies['genres'].str.contains(genre)]
                
                if not example_movies.empty:
                    st.dataframe(example_movies[['title', 'genres']].head(10), use_container_width=True)
                else:
                    st.write("No examples found.")
        else:
            st.write(f"No combinations of {selected_combo_size} genres found.")
    
    else:
        # If direct CSV access doesn't work, try using the getGenres method
        if hasattr(ml, 'getGenres'):
            genres = ml.getGenres()
            
            if genres:
                st.write("Note: Using binary genre representation as direct genre data is not available")
                st.write("According to MovieLens documentation, the genres are (in order):")
                st.write(", ".join(genre_names))
                
                # Convert first few to a dataframe for display
                genre_items = list(genres.items())[:100]  # Limit to 100 for performance
                
                # Create a sample dataframe
                sample_movies = []
                for movie_id, genre_list in genre_items:
                    if movie_id in ml.movieID_to_name:
                        # Convert binary indicators to genre names
                        movie_genres = []
                        for i, val in enumerate(genre_list):
                            if val == 1 and i < len(genre_names):
                                movie_genres.append(genre_names[i])
                        
                        genre_str = ', '.join(movie_genres) if movie_genres else "(no genres listed)"
                        sample_movies.append({
                            'movieId': movie_id,
                            'title': ml.movieID_to_name[movie_id],
                            'genres': genre_str
                        })
                
                genre_df = pd.DataFrame(sample_movies)
                
                st.write(f"Genre information available for {len(genres)} movies")
                st.subheader("Sample Movie Genres")
                st.dataframe(genre_df, use_container_width=True)
        else:
            st.error("Genre information not available")

# Tab 5: Years
with tabs[4]:
    st.header("Movie Release Years")
    
    # Get years
    if hasattr(ml, 'getYears'):
        years = ml.getYears()
        
        if years:
            # Convert to dataframe
            years_df = pd.DataFrame(list(years.items()), columns=['movieId', 'year'])
            
            # Add titles
            years_df['title'] = years_df['movieId'].map(ml.movieID_to_name)
            
            # Display statistics
            min_year = years_df['year'].min()
            max_year = years_df['year'].max()
            st.write(f"Years range from {min_year} to {max_year}")
            st.write(f"Total movies with year data: {len(years_df)}")
            
            # Display sample
            st.subheader("Sample Movies with Year Data")
            st.dataframe(years_df[['year', 'title']].sample(10), use_container_width=True)
            
            # Plot year distribution
            st.subheader("Movies by Decade")
            years_df['decade'] = (years_df['year'] // 10) * 10
            decade_counts = years_df['decade'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(decade_counts.index.astype(str), decade_counts.values)
            ax.set_xlabel('Decade')
            ax.set_ylabel('Number of Movies')
            ax.set_title('Movies by Decade')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Search by year
            st.subheader("Search Movies by Year")
            selected_year = st.slider("Select Year", min_year, max_year, min_year + (max_year - min_year) // 2)

            year_movies_ids = years_df[years_df['year'] == selected_year]['movieId'].tolist()
            year_movies = years_df[years_df['year'] == selected_year][['title']].sort_values('title')

            if not year_movies.empty:
                st.write(f"Found {len(year_movies)} movies from {selected_year}")
                st.dataframe(year_movies, use_container_width=True)
                
                # Get genre distribution for this year's movies
                movies_path = os.path.join(parent_dir, 'ml-latest-small', 'movies.csv')
                if os.path.exists(movies_path):
                    # Load movies with genres
                    movies_with_genres = pd.read_csv(movies_path)
                    
                    # Filter to only selected year's movies
                    year_movies_with_genres = movies_with_genres[movies_with_genres['movieId'].isin(year_movies_ids)]
                    
                    # Extract all genres from this year's movies
                    year_genres = []
                    for genres_str in year_movies_with_genres['genres']:
                        if isinstance(genres_str, str):
                            year_genres.extend(genres_str.split('|'))
                    
                    if year_genres:
                        # Count each genre for this year
                        year_genre_counts = pd.Series(year_genres).value_counts()
                        
                        # Plot the genre distribution as vertical bars
                        st.subheader(f"Genre Distribution in {selected_year}")
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # Sort by count for better visualization
                        year_genre_counts = year_genre_counts.sort_values(ascending=False)
                        
                        # Use vertical bar chart
                        bars = ax.bar(year_genre_counts.index, year_genre_counts.values, color='#5AC8FA')  # Electric blue from your palette
                        
                        # Add count labels above each bar
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    f'{int(height)}', ha='center', va='bottom')
                        
                        ax.set_xlabel('Genre')
                        ax.set_ylabel('Number of Movies')
                        ax.set_title(f'Genres of Movies Released in {selected_year}')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Compare to overall genre distribution
                        if st.checkbox(f"Compare {selected_year} genres to overall distribution"):
                            # Get overall genre counts
                            all_genres = []
                            for genres_str in movies_with_genres['genres']:
                                if isinstance(genres_str, str):
                                    all_genres.extend(genres_str.split('|'))
                            overall_genre_counts = pd.Series(all_genres).value_counts()
                            
                            # Calculate percentage of each genre for this year
                            # compared to overall percentage
                            all_genres_in_year = sum(year_genre_counts.values)
                            all_genres_total = sum(overall_genre_counts.values)
                            
                            # Create comparison dataframe
                            comparison = pd.DataFrame({
                                'Genre': year_genre_counts.index,
                                f'{selected_year} Count': year_genre_counts.values,
                                f'{selected_year} %': year_genre_counts.values / all_genres_in_year * 100,
                                'Overall %': [overall_genre_counts.get(g, 0) / all_genres_total * 100 
                                            for g in year_genre_counts.index],
                            })
                            
                            # Calculate difference from average
                            comparison['Difference'] = comparison[f'{selected_year} %'] - comparison['Overall %']
                            comparison = comparison.sort_values('Difference', ascending=False)
                            
                            # Plot difference from average
                            st.subheader(f"Genre Trends in {selected_year} (Compared to Overall Average)")
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            # Color bars based on whether they're above or below average
                            colors = ['#00F2C3' if x > 0 else '#FF3B30' for x in comparison['Difference']]
                            
                            bars = ax.bar(comparison['Genre'], comparison['Difference'], color=colors)
                            
                            # Add baseline at 0
                            ax.axhline(y=0, color='#BBBBBB', linestyle='-', alpha=0.3)
                            
                            ax.set_xlabel('Genre')
                            ax.set_ylabel('Percentage Difference from Average')
                            ax.set_title(f'How {selected_year} Differs from Overall Genre Distribution')
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Show the data
                            st.write("Comparison Data:")
                            st.dataframe(comparison.round(2), use_container_width=True)
            else:
                st.write(f"No movies found from {selected_year}")
    else:
        st.error("getYears method not available in the MovieLens object")

# Tab 6: Evaluation Data
with tabs[5]:
    st.header("Additional Details")
    
    # Show information about the evaluation data
    st.write(f"Type: {type(evaluationData).__name__}")
    
    if hasattr(evaluationData, 'raw_ratings'):
        st.write(f"Number of raw ratings: {len(evaluationData.raw_ratings)}")
        
        # Display sample ratings
        st.subheader("Sample Raw Ratings")
        
        sample_ratings = []
        for i, (user, item, rating, timestamp) in enumerate(evaluationData.raw_ratings[:10]):
            movie_name = ml.movieID_to_name.get(int(item), "Unknown")
            readable_time = pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')
            
            sample_ratings.append({
                "User ID": user,
                "Movie": movie_name,
                "Rating": rating,
                "Timestamp": readable_time
            })

        # Convert to DataFrame and display as table
        sample_df = pd.DataFrame(sample_ratings)
        st.dataframe(sample_df, use_container_width=True)
        
        # Build trainset
        if hasattr(evaluationData, 'build_full_trainset'):
            trainset = evaluationData.build_full_trainset()
            
            st.subheader("Training Data Statistics")
            st.write(f"Number of users: {trainset.n_users}")
            st.write(f"Number of movies: {trainset.n_items}")
            st.write(f"Number of ratings: {trainset.n_ratings}")
            st.write(f"Rating scale: {trainset.rating_scale}")
            
            # Display rating distribution
            all_ratings = [r for (_, _, r) in trainset.all_ratings()]
            
            st.subheader("Rating Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(all_ratings, bins=9, rwidth=0.8)
            ax.set_xlabel('Rating Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Ratings')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show user activity
            st.subheader("User Activity")
            
            # Get number of ratings per user
            user_counts = {}
            for u_id in trainset.all_users():
                user_counts[u_id] = len(trainset.ur[u_id])
            
            # Convert to dataframe
            user_df = pd.DataFrame(list(user_counts.items()), columns=['user_id', 'rating_count'])
            user_df = user_df.sort_values('rating_count', ascending=False)
            
            # Display top users
            st.write("Most Active Users")
            
            top_users = user_df.head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(top_users['user_id'].astype(str), top_users['rating_count'])
            ax.set_xlabel('User ID')
            ax.set_ylabel('Number of Ratings')
            ax.set_title('Most Active Users')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show item popularity
            st.subheader("Movie Popularity")
            
            # Get number of ratings per movie
            item_counts = {}
            for i_id in trainset.all_items():
                item_counts[i_id] = len(trainset.ir[i_id])
            
            # Convert to dataframe
            item_df = pd.DataFrame(list(item_counts.items()), columns=['item_id', 'rating_count'])
            item_df = item_df.sort_values('rating_count', ascending=False)
            
            # Add movie titles
            item_df['title'] = item_df['item_id'].apply(lambda x: ml.movieID_to_name.get(int(trainset.to_raw_iid(x)), "Unknown"))
            
            # Display top movies
            st.write("Most Rated Movies")
            
            top_items = item_df.head(10)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.barh(top_items['title'].str[:30], top_items['rating_count'])
            ax.set_xlabel('Number of Ratings')
            ax.set_ylabel('Movie')
            ax.set_title('Most Rated Movies')
            ax.invert_yaxis()  # To show highest count at top
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.error("raw_ratings not available in the evaluationData object")


# Content for the recommendations tab
with tabs[6]:
    st.header("Movie Recommendation System")
    st.markdown("First, select a user to see what they've watched, then get personalized recommendations.")

    # User selection
    user_ids = list(range(1, 672))  # MovieLens typically has users 1-671
    selected_user = st.selectbox("Select User ID:", user_ids, index=0)
    
    # Show the user's movies first
    if selected_user:
        # Get user ratings
        user_ratings = ml.getUserRatings(selected_user)
        
        if isinstance(user_ratings, dict):
            # Convert to dataframe
            user_df = pd.DataFrame(list(user_ratings.items()), columns=['movieId', 'rating'])
        elif isinstance(user_ratings, list):
            # If it's a list of tuples
            user_df = pd.DataFrame(user_ratings, columns=['movieId', 'rating'])
        else:
            user_df = pd.DataFrame()
        
        if not user_df.empty:
            # Join with movie titles
            user_df['title'] = user_df['movieId'].map(ml.movieID_to_name)
            
            # Sort by rating
            user_df = user_df.sort_values('rating', ascending=False)
            
            # Show the user's ratings
            st.subheader(f"User {selected_user} has rated {len(user_df)} movies")
            st.dataframe(user_df[['title', 'rating']], use_container_width=True)
        else:
            st.warning(f"No ratings found for user {selected_user}")
    
    # After showing the dataframe of user ratings, add genre and year distribution plots
    if not user_df.empty:
        try:
            # Load movies data to get genre information
            movies_path = os.path.join(parent_dir, 'ml-latest-small', 'movies.csv')
            if os.path.exists(movies_path):
                # Load movies with genres
                movies_with_genres = pd.read_csv(movies_path)
                
                # Merge with user's ratings
                user_movies_with_genres = pd.merge(
                    user_df, 
                    movies_with_genres, 
                    on='movieId', 
                    how='left',
                    suffixes=('', '_full')
                )
                
                # ===================== GENRE DISTRIBUTION =====================
                st.subheader(f"Genre Distribution for User {selected_user}")

                # Extract all genres that the user has rated
                user_genres = []
                for genres_str in user_movies_with_genres['genres']:
                    if isinstance(genres_str, str):  # Check if it's a valid string
                        user_genres.extend(genres_str.split('|'))

                if user_genres:
                    # Count each genre
                    genre_counts = pd.Series(user_genres).value_counts()
                    
                    # Plot the genre distribution
                    fig, ax = plt.subplots(figsize=(14, 8))  # Wider figure for vertical bars
                    
                    # Sort by count for better visualization
                    genre_counts = genre_counts.sort_values(ascending=False)
                    
                    # Use vertical bar chart
                    bars = ax.bar(genre_counts.index, genre_counts.values, color='#FF9500')  # Neon orange from your palette
                    
                    # Set axis labels (switched from horizontal)
                    ax.set_ylabel('Number of Movies')
                    ax.set_xlabel('Genre')
                    ax.set_title(f'Genres Watched by User {selected_user}')
                    
                    # Add count labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                f'{int(height)}', ha='center', va='bottom')
                    
                    # Rotate x-axis labels for better readability
                    plt.xticks(rotation=45, ha='right')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # ===================== YEAR DISTRIBUTION =====================
                    st.subheader(f"Movie Release Year Distribution for User {selected_user}")

                    # Extract years from movie titles
                    def extract_year(title):
                        match = re.search(r'\((\d{4})\)', title)
                        if match:
                            return int(match.group(1))
                        return None

                    # Apply year extraction to the titles
                    user_movies_with_genres['year'] = user_movies_with_genres['title'].apply(extract_year)

                    # Count movies by year
                    year_counts = user_movies_with_genres['year'].value_counts().sort_index()

                    # Filter out None values
                    if None in year_counts:
                        year_counts = year_counts.drop(None)

                    # Plot year distribution
                    if not year_counts.empty:
                        # Define the BubbleChart class for force-directed bubble layout
                        class BubbleChart:
                            def __init__(self, area, bubble_spacing=0):
                                """
                                Setup for bubble collapse.
                                """
                                area = np.asarray(area)
                                r = np.sqrt(area / np.pi)

                                self.bubble_spacing = bubble_spacing
                                self.bubbles = np.ones((len(area), 4))
                                self.bubbles[:, 2] = r
                                self.bubbles[:, 3] = area
                                self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
                                self.step_dist = self.maxstep / 2

                                # calculate initial grid layout for bubbles
                                length = np.ceil(np.sqrt(len(self.bubbles)))
                                grid = np.arange(length) * self.maxstep
                                gx, gy = np.meshgrid(grid, grid)
                                self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
                                self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

                                self.com = self.center_of_mass()

                            def center_of_mass(self):
                                return np.average(
                                    self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
                                )

                            def center_distance(self, bubble, bubbles):
                                return np.hypot(bubble[0] - bubbles[:, 0],
                                            bubble[1] - bubbles[:, 1])

                            def outline_distance(self, bubble, bubbles):
                                center_distance = self.center_distance(bubble, bubbles)
                                return center_distance - bubble[2] - \
                                    bubbles[:, 2] - self.bubble_spacing

                            def check_collisions(self, bubble, bubbles):
                                distance = self.outline_distance(bubble, bubbles)
                                return len(distance[distance < 0])

                            def collides_with(self, bubble, bubbles):
                                distance = self.outline_distance(bubble, bubbles)
                                idx = np.argmin(distance)
                                return np.array([idx])

                            def collapse(self, n_iterations=50):
                                """
                                Move bubbles to the center of mass.
                                """
                                for _i in range(n_iterations):
                                    moves = 0
                                    for i in range(len(self.bubbles)):
                                        rest_bub = np.delete(self.bubbles, i, 0)
                                        if len(rest_bub) == 0:
                                            continue
                                            
                                        # try to move directly towards the center of mass
                                        # direction vector from bubble to the center of mass
                                        dir_vec = self.com - self.bubbles[i, :2]

                                        # shorten direction vector to have length of 1
                                        if dir_vec.dot(dir_vec) > 0:  # Avoid division by zero
                                            dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                                            # calculate new bubble position
                                            new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                                            new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                                            # check whether new bubble collides with other bubbles
                                            if not self.check_collisions(new_bubble, rest_bub):
                                                self.bubbles[i, :] = new_bubble
                                                self.com = self.center_of_mass()
                                                moves += 1
                                            else:
                                                # try to move around a bubble that you collide with
                                                # find colliding bubble
                                                for colliding in self.collides_with(new_bubble, rest_bub):
                                                    # calculate direction vector
                                                    dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                                                    if dir_vec.dot(dir_vec) > 0:  # Avoid division by zero
                                                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                                                        # calculate orthogonal vector
                                                        orth = np.array([dir_vec[1], -dir_vec[0]])
                                                        # test which direction to go
                                                        new_point1 = (self.bubbles[i, :2] + orth * self.step_dist)
                                                        new_point2 = (self.bubbles[i, :2] - orth * self.step_dist)
                                                        dist1 = self.center_distance(self.com, np.array([new_point1]))
                                                        dist2 = self.center_distance(self.com, np.array([new_point2]))
                                                        new_point = new_point1 if dist1 < dist2 else new_point2
                                                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                                                        if not self.check_collisions(new_bubble, rest_bub):
                                                            self.bubbles[i, :] = new_bubble
                                                            self.com = self.center_of_mass()

                                    if moves / max(1, len(self.bubbles)) < 0.1:
                                        self.step_dist = self.step_dist / 2

                            def plot(self, ax, labels, counts, cmap='plasma'):
                                """
                                Draw the bubble plot.
                                """
                                # Generate colors based on counts
                                norm = plt.Normalize(min(counts), max(counts))
                                colors = plt.cm.get_cmap(cmap)(norm(counts))
                                
                                # Draw the bubbles
                                for i in range(len(self.bubbles)):
                                    circ = plt.Circle(
                                        self.bubbles[i, :2], self.bubbles[i, 2], 
                                        color=colors[i], alpha=0.7, edgecolor='white')
                                    ax.add_patch(circ)
                                    
                                    # Add the label inside the bubble
                                    fontsize = 25
                                    ax.text(self.bubbles[i, 0], self.bubbles[i, 1], 
                                        f"{labels[i]}\n({counts[i]})",
                                        horizontalalignment='center', 
                                        verticalalignment='center',
                                        fontweight='bold',
                                        color='white',
                                        fontsize=fontsize)
                                
                                # Return the colors for colorbar
                                return colors, norm
                        
                        # Set up the plot
                        fig, ax = plt.subplots(figsize=(14, 10), subplot_kw=dict(aspect="equal"))
                        
                        # Extract data
                        years = list(year_counts.index)
                        counts = list(year_counts.values)
                        
                        # Create bubble chart with the counts as areas (multiplied for better visibility)
                        bubble_areas = np.array(counts) * 25  # Scale factor
                        
                        # Initialize and calculate bubble positions
                        bubble_chart = BubbleChart(area=bubble_areas, bubble_spacing=0.1)
                        bubble_chart.collapse()
                        
                        # Plot the bubbles with the years as labels
                        colors, norm = bubble_chart.plot(ax, years, counts)
                        
                        # Set up the axes and title
                        ax.axis("on")  # Turn off the axis
                        ax.set_title(f'Movie Release Years Watched by User {selected_user}')
                        ax.relim()
                        ax.autoscale_view()
                        
                        # Add colorbar
                        sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
                        sm.set_array([])
                        cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
                        cbar.set_label('Number of Movies')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # ===================== YEAR BY DECADE =====================
                        # Also add a decade-based histogram for a different view
                        st.subheader(f"Movies by Decade for User {selected_user}")
                        
                        # Group by decade
                        user_movies_with_genres['decade'] = (user_movies_with_genres['year'] // 10) * 10
                        decade_counts = user_movies_with_genres['decade'].value_counts().sort_index()
                        
                        # Remove None values
                        if None in decade_counts:
                            decade_counts = decade_counts.drop(None)
                        
                        # Plot decade distribution
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.bar(decade_counts.index.astype(str), decade_counts.values, color='#00F2C3')  # Bright teal
                        
                        # Add count labels on top of bars
                        for i, v in enumerate(decade_counts.values):
                            ax.text(i, v + 0.5, str(v), ha='center')
                        
                        ax.set_xlabel('Decade')
                        ax.set_ylabel('Number of Movies')
                        ax.set_title(f'Movies by Decade Watched by User {selected_user}')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.write("No year data available for this user's movies.")
                else:
                    st.write("No genre information available for this user's movies.")
            else:
                st.write("Could not load movie metadata for visualization.")
        except Exception as e:
            st.error(f"Error creating visualizations: {str(e)}")

    # Find all available model files
    st.subheader("Generate Movie Recommendations")

    # Define model directory paths
    model_dirs = {
        'ItemKNN': os.path.join(parent_dir, 'models', '1_ItemBasedCollaborativeFiltering'),
        'UserKNN': os.path.join(parent_dir, 'models', '1_UserBasedCollaborativeFiltering'),
        'MatrixFactorization': os.path.join(parent_dir, 'models', '2_MatrixFactorization'),
        'ContentBased': os.path.join(parent_dir, 'models', '3_ContentBased')
    }

    # Search for all model files
    all_model_files = []
    for category, dir_path in model_dirs.items():
        if os.path.exists(dir_path):
            for filepath in glob.glob(os.path.join(dir_path, '*.pkl')):
                model_name = os.path.basename(filepath)
                # Create a display name from the filename
                display_name = model_name.replace('adapted_', '').replace('_model_', ' ').replace('.pkl', '')
                
                # Clean up the display name
                if 'knn' in display_name:
                    if 'item' in display_name:
                        display_name = f"Item KNN ({display_name.split('item_knn_')[1].capitalize()})"
                    else:
                        display_name = f"User KNN ({display_name.split('user_knn_')[1].capitalize()})"
                elif 'als' in display_name:
                    display_name = "ALS (Alternating Least Squares)"
                elif 'nmf' in display_name:
                    display_name = "NMF (Non-negative Matrix Factorization)"
                elif 'pmf' in display_name:
                    display_name = "PMF (Probabilistic Matrix Factorization)"
                elif 'svdpp' in display_name:
                    display_name = "SVD++ (Enhanced SVD)"
                elif 'svd' in display_name:
                    display_name = "SVD (Singular Value Decomposition)"
                elif 'tf_idf' in display_name:
                    display_name = "TF-IDF (Content-Based)"
                
                all_model_files.append((display_name, filepath, category))

    # Sort models by category and name
    all_model_files.sort(key=lambda x: (x[2], x[0]))

    # Extract just the display names for the selectbox
    model_display_names = [model[0] for model in all_model_files]
    model_filepaths = {model[0]: model[1] for model in all_model_files}

    # Algorithm selection
    if model_display_names:
        model_choice = st.selectbox(
            "Select recommendation algorithm:", 
            model_display_names,
            index=0  # Default to the first option
        )
        
        # Get selected model path
        selected_model_path = model_filepaths[model_choice]
    else:
        st.error("No recommendation models found in the models directory.")
        selected_model_path = None
    
    # Number of recommendations
    num_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
    
    # Get recommendations button
    if st.button("Generate Recommendations", key="generate_recs"):
        try:
            with st.spinner("Finding the best movies for you..."):
                with open(selected_model_path, 'rb') as f:
                    model = pickle.load(f)
                
                # Generate recommendations
                user_id = selected_user
                
                # Get movies the user has already rated
                if isinstance(user_ratings, dict):
                    rated_movie_ids = set(user_ratings.keys())
                elif isinstance(user_ratings, list):
                    rated_movie_ids = set(item for item, _ in user_ratings)
                else:
                    rated_movie_ids = set()
                
                # Get all movie IDs
                all_movie_ids = set(ml.movieID_to_name.keys())
                
                # Find movies the user hasn't rated
                unrated_movie_ids = list(all_movie_ids - rated_movie_ids)
                
                # Get predictions for unrated movies
                predictions = []
                for movie_id in unrated_movie_ids[:200]:  # Limit for performance
                    try:
                        prediction = model.estimate(user_id, movie_id)
                        predictions.append((movie_id, prediction))
                    except:
                        continue
                
                # Sort and get top N
                recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:num_recommendations]
                
                # Display results
                st.success(f"Top {num_recommendations} recommended movies for User {user_id}:")

                # Create a table of recommendations
                table_data = []

                for i, (movie_id, rating) in enumerate(recommendations):
                    movie_name = ml.movieID_to_name.get(int(movie_id), "Unknown")
                    
                    # Get genre information
                    genres = "Not available"
                    try:
                        # Try to get genre information if available
                        movies_path = os.path.join(parent_dir, 'ml-latest-small', 'movies.csv')
                        if os.path.exists(movies_path):
                            movies_df = pd.read_csv(movies_path)
                            movie_info = movies_df[movies_df['movieId'] == movie_id]
                            if not movie_info.empty:
                                genres = movie_info['genres'].values[0]
                    except:
                        pass
                    
                    # Add data to table
                    table_data.append({
                        "Rank": i+1,
                        "Movie": movie_name,
                        "Predicted Rating": f"{rating:.2f}/5.0",
                        "Genres": genres
                    })

                # Create DataFrame and display as table
                if table_data:
                    recommendations_df = pd.DataFrame(table_data)
                    st.dataframe(recommendations_df, use_container_width=True)
                    
                    # Create CSV data
                    csv = recommendations_df.to_csv(index=False).encode('utf-8')
                    
                    # Display the download button directly (not nested in another button)
                    st.download_button(
                        label="Download Recommendations as CSV",
                        data=csv,
                        file_name=f"movie_recommendations_user_{user_id}_{model_choice.replace(' ', '_').replace('(', '').replace(')', '')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No recommendations could be generated.")
                
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            st.exception(e)

# Add a footer
st.markdown("---")
st.caption("MovieLens Data Explorer - Created by Alexandru-Flavius Huc")