# Item-Based Collaborative Filtering Algorithms Documentation

![ItemBasedColloaborativeFiltering](../images/ItemBasedColloaborativeFiltering.png)

## Overview

The `1_ItemBasedCollaborativeFiltering` folder contains implementations of item-based collaborative filtering algorithms for movie recommendation. These algorithms predict user preferences for movies based on the similarity between items (movies) rather than users.

## Algorithm: ItemKNN

### Description
Item-KNN is an item-based collaborative filtering algorithm that recommends items similar to those a user has previously rated highly. It works by finding the k most similar items to a target item and using the user's ratings on these similar items to predict their rating for the target item.

### Implementation Classes

#### `PureItemKNN`
A pure Python implementation of item-based collaborative filtering using k-nearest neighbors.

##### Parameters
- `k` (int, default=40): Number of neighbors to use for prediction
- `sim_options` (dict): Options for similarity calculation
  - `name`: 'pearson' or 'cosine'
  - `user_based`: False (to ensure item-based approach)

##### Key Methods
- `fit(ratings_data)`: Trains the algorithm on (user_id, item_id, rating) tuples
- `predict(user_id, item_id)`: Predicts rating for a specific user-item pair
- `_compute_similarity(item1, item2)`: Calculates similarity between two items
- `_compute_cosine_similarity(ratings1, ratings2, common_users)`: Implements cosine similarity
- `_compute_pearson_similarity(ratings1, ratings2, common_users, item1, item2)`: Implements Pearson correlation

##### Implementation Details
1. **Similarity Matrix Computation**:
   - Builds a complete item-item similarity matrix
   - Supports both Pearson correlation and cosine similarity
   - Only considers items with at least 2 common users for meaningful similarity

2. **Rating Prediction**:
   - Uses weighted average of ratings from similar items
   - Applies mean-centering to account for item rating biases
   - Falls back to item mean rating when no similar items found

3. **Optimization Techniques**:
   - Stores ratings by item for efficient similarity computation
   - Pre-computes item means to speed up Pearson correlation
   - Uses heap queue for efficient top-k neighbors selection

#### `AdaptedItemKNN`
An adapter class that makes `PureItemKNN` compatible with Surprise's evaluation framework.

##### Key Methods
- `fit(trainset)`: Converts Surprise's trainset to the format needed by `PureItemKNN`
- `estimate(u, i)`: Wrapper around `PureItemKNN.predict()` for the Surprise framework

### Variants

#### ItemKNN-Pearson
Uses Pearson correlation coefficient to measure item similarity. This measures the linear correlation between item rating patterns while accounting for different rating scales and biases between items.

**Advantages**:
- Accounts for user rating bias
- Works well when users have different rating scales
- More robust to outliers than cosine similarity

#### ItemKNN-Cosine
Uses cosine similarity to measure item similarity. This measures the cosine of the angle between item rating vectors, focusing on the direction rather than magnitude.

**Advantages**:
- Simpler to understand and implement
- Works well for sparse data
- Computation is slightly faster than Pearson

### Performance Considerations

1. **Time Complexity**:
   - Building similarity matrix: O(n²m), where n is the number of items and m is the average number of users per item
   - Prediction: O(k log u), where k is the number of neighbors and u is the number of items the user has rated

2. **Space Complexity**:
   - O(n²) for the similarity matrix
   - O(nm) for the ratings data

3. **Scalability**:
   - Similarity matrix computation is the main bottleneck
   - Implementation includes progress tracking for large datasets
   - Matrix computation can be parallelized (not implemented in current version)

### Usage Example

```python
# Initialize the algorithm
item_knn = PureItemKNN(k=40, sim_options={'name': 'pearson', 'user_based': False})

# Train the algorithm
item_knn.fit(ratings_data)

# Make predictions
prediction = item_knn.predict(user_id=123, item_id=456)

# For use with the evaluation framework
adapted_knn = AdaptedItemKNN(k=40, sim_options={'name': 'pearson', 'user_based': False})
evaluator.AddAlgorithm(adapted_knn, "ItemKNN-Pearson")