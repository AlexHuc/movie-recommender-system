# Evaluation Framework Documentation

This directory contains the evaluation framework used to test and compare different recommendation algorithms. The framework provides standardized methods to measure algorithm performance across multiple metrics.

## File Descriptions

### `Evaluator.py`
The main evaluation engine that coordinates the testing process. It:
- Takes algorithms and datasets as input
- Runs all algorithms against the same dataset
- Collects performance metrics including RMSE, MAE, and coverage
- Generates comparison reports across algorithms
- Provides functionality to sample Top-N recommendations

### `EvaluatedAlgorithm.py`
Wraps recommendation algorithms with evaluation capabilities:
- Adapts algorithms to a common interface
- Abstracts algorithm-specific details from the evaluation process
- Handles computation of metrics like RMSE and MAE
- Supports both rating prediction and Top-N recommendation evaluation

### `MovieLens.py`
Handles data loading and preprocessing for the MovieLens dataset:
- Loads rating data, movie metadata, and tags
- Creates training and testing splits
- Computes movie popularity rankings
- Extracts and normalizes content features (genres, years)
- Provides helper methods for accessing movie information

### `RecommenderMetrics.py`
Contains implementations of different evaluation metrics:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Hit Rate for Top-N recommendations
- Average Reciprocal Hit Rank (ARHR)
- Coverage metrics for recommendation diversity
- Novelty and diversity measurements
- Utility functions for Top-N recommendation evaluation

### `EvaluationData.py`
Manages evaluation dataset splitting and organization:
- Splits data into training and testing sets
- Creates leave-one-out cross-validation folds
- Organizes anti-test-set for Top-N evaluation
- Provides consistent data access methods for algorithms

## Usage Example

```python
# Create evaluator with data and rankings
evaluator = Evaluator(data, rankings)

# Add algorithms to test
evaluator.AddAlgorithm(algorithm1, "Algorithm 1")
evaluator.AddAlgorithm(algorithm2, "Algorithm 2")

# Run evaluation
evaluator.Evaluate(doTopN=True)

# Sample top recommendations
evaluator.SampleTopNRecs(ml)