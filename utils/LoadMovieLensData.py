from EvaluationFramework.MovieLens import MovieLens

def LoadMovieLensData():
    """
    Loads the MovieLens dataset and computes popularity rankings.
    
    This utility function centralizes the data loading process used across
    different recommendation algorithms in the project.
    
    Returns:
        tuple: A 3-element tuple containing:
            - ml (MovieLens): The MovieLens data handler object
            - data (Dataset): The loaded dataset with ratings
            - rankings (dict): Dictionary mapping movie IDs to popularity ranks
    """
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)