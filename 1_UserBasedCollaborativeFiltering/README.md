# User-Based Collaborative Filtering Algorithms Documentation

![UserBasedColloaborativeFiltering](../images/UserBasedColloaborativeFiltering.png)

## Overview

The `1_UserBasedCollaborativeFiltering` folder contains implementations of user-based collaborative filtering algorithms for movie recommendation. These algorithms predict item ratings and generate recommendations by identifying users with similar tastes to the target user.

## Algorithm: UserKNN

### Description
User-KNN (K-Nearest Neighbors) is a user-based collaborative filtering algorithm that recommends items liked by similar users. It works by finding the k most similar users to a target user and then using their ratings to predict the target user's preferences for unseen items.

### Implementation Classes

#### `PureUserKNN`
A pure Python implementation of user-based collaborative filtering using k-nearest neighbors.

##### Parameters
- `k` (int, default=40): Number of nearest neighbors to consider for predictions
- `sim_options` (dict): Options for similarity calculation
  - `name`: 'pearson' or 'cosine'
  - `user_based`: True (to ensure user-based approach)

##### Key Methods
- `fit(ratings_data)`: Trains the algorithm on (user_id, item_id, rating) tuples
- `predict(user_id, item_id)`: Predicts rating for a specific user-item pair
- `_compute_similarity(user1, user2)`: Calculates similarity between two users
- `_compute_cosine_similarity(ratings1, ratings2, common_items)`: Implements cosine similarity
- `_compute_pearson_similarity(ratings1, ratings2, common_items, user1, user2)`: Implements Pearson correlation

##### Implementation Details
1. **Similarity Matrix Computation**:
   - Builds a complete user-user similarity matrix
   - Supports both Pearson correlation and cosine similarity
   - Only considers users with at least 2 common items for meaningful similarity
   - Uses symmetry to reduce computation time (compute sim(u,v) once, not twice)

2. **Rating Prediction**:
   - Uses weighted average of ratings from similar users
   - Applies mean-centering to account for user rating biases
   - Only considers positively correlated users for recommendations
   - Falls back to user mean rating when no neighbors found

3. **Optimization Techniques**:
   - Maintains an inverted index (item → users who rated it) for faster lookups
   - Pre-computes user means to speed up Pearson correlation
   - Uses heap queue for efficient top-k neighbors selection
   - Progress tracking for similarity matrix computation

#### `AdaptedUserKNN`
An adapter class that wraps `PureUserKNN` to make it compatible with Surprise's evaluation framework.

##### Key Methods
- `fit(trainset)`: Converts Surprise's trainset to the format needed by `PureUserKNN`
- `estimate(u, i)`: Wrapper around `PureUserKNN.predict()` for the Surprise framework

### Variants

#### UserKNN-Pearson
Uses Pearson correlation coefficient to measure user similarity. This measures the linear correlation between user rating patterns while accounting for different rating scales.

**Advantages**:
- Accounts for differences in user rating scales and biases
- More accurate when users have varying rating tendencies
- Handles the "tough rater" and "easy rater" problem well

#### UserKNN-Cosine
Uses cosine similarity to measure user similarity. This measures the cosine of the angle between user rating vectors, focusing on the direction rather than magnitude.

**Advantages**:
- Simpler and computationally more efficient
- Can work well when users have consistent rating scales
- Sometimes better with very sparse data

### Performance Considerations

1. **Time Complexity**:
   - Building similarity matrix: O(n²m), where n is the number of users and m is the average number of items per user
   - Prediction: O(k log n), where k is the number of neighbors and n is the number of users who rated the item

2. **Space Complexity**:
   - O(n²) for the similarity matrix
   - O(nm) for the ratings data
   - O(m) for the inverted index, where m is the number of items

3. **Scalability**:
   - Similarity matrix computation is the main bottleneck
   - Implementation includes progress tracking for large datasets
   - More efficient for datasets with many items but fewer users

### Usage Example

```python
# Initialize the algorithm
user_knn = PureUserKNN(k=40, sim_options={'name': 'pearson', 'user_based': True})

# Train the algorithm
user_knn.fit(ratings_data)

# Make predictions
prediction = user_knn.predict(user_id=123, item_id=456)

# For use with the evaluation framework
adapted_knn = AdaptedUserKNN(k=40, sim_options={'name': 'pearson', 'user_based': True})
evaluator.AddAlgorithm(adapted_knn, "UserKNN-Pearson")