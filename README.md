# Movie Recommendation System

![MovieRecommendationSystem](images/MovieRecommendationSystem.png)

## Overview
This project implements a machine learning-based movie recommendation systems using the MovieLens dataset. The system analyzes user preferences and movie features to predict ratings and recommend relevant movies to users. All algorithms are implemented from scratch in pure Python to demonstrate the underlying mathematics and concepts of recommendation systems.

## Project Goals
- Develop accurate prediction models for user movie preferences
- Compare performance of different recommendation algorithms
- Create a system that can suggest personalized movie recommendations
- Analyze user behavior patterns to improve prediction accuracy
- Implement machine learning algorithms using only core Python (no ML libraries)
- Evaluate algorithm performance using a custom EvaluationFramework

## Dataset
The project uses the MovieLens dataset provided by GroupLens Research. This dataset includes:
- 100,004 ratings on 9,125 movies by 671 users
- Movie metadata including titles, genres, and tags
- User rating history and tagging behavior

## Implemented Algorithms

### 1. Collaborative Filtering
**User-Based Collaborative Filtering**
- K-Nearest Neighbors (KNN)
- Cosine similarity / Pearson correlation

**Item-Based Collaborative Filtering**
- KNN with item similarity
- Matrix factorization

### 2. Matrix Factorization Techniques
- Singular Value Decomposition (SVD)
- SVD++
- Non-negative Matrix Factorization (NMF)
- Alternating Least Squares (ALS)
- Probabilistic Matrix Factorization (PMF)

### 3. Content-Based Filtering
- Logistic Regression
- Decision Trees
- Naive Bayes
- Support Vector Machines (SVM)
- k-NN with feature vectors
- TF-IDF + Cosine similarity (for text-based content)

### Evaluation of the algorithms

| Agorithms                       | RMSE   | MAE       |
|---------------------------------|--------|-----------|
| Random                          | 1.4385 |    1.1478 |
| Collaborative Filtering         |--------|-----------|
| ItemKNN-Pearson                 | 0.9287 |    0.7109 |   
| ItemKNN-Cosine                  | 0.9263 |    0.7103 |   
| UserKNN-Pearson                 | 0.9231 |    0.7056 |   
| UserKNN-Cosine                  | 0.9259 |    0.7116 |
| Matrix Factorization Techniques |--------|-----------|
| ALS                             | 2.7220 |    2.5019 | 
| NMF                             | 1.0133 |    0.7930 | 
| PMF                             | 1.0549 |    0.8546 |  
| SVD                             | 0.9187 |    0.7112 |   
| SVD-Large                       | 0.9195 |    0.7085 |  
| SVD++                           | 0.9187 |    0.7112 |
| Content-Based Filtering         |--------|-----------|
| DecisionTree                    | 1.0742 |    0.8115 | 
| PureKNN                         | 0.9375 |    0.7264 | 
| LogisticRegression              | 1.1744 |    0.9179 | 
| NaiveBayes                      | 1.1283 |    0.8338 | 
| PureSVM                         | 0.9870 |    0.7534 | 
| TF-IDF                          | 0.9553 |    0.7421 | 

## Technologies
- Python for data processing and pure algorithm implementation
- Pandas and NumPy for data manipulation
- Custom EvaluationFramework builded on surprise library for testing and comparing algorithms
- Matplotlib/Seaborn for data visualization

## Getting Started
1. Clone this repository
2. Install dependencies from requirements.txt
3. Run the preprocessing scripts to prepare the data
4. Train models using the provided notebooks
5. Evaluate model performance with test datasets
6. Generate recommendations using the trained models

## Project Structure
- `/ml-latest-small`: Contains the MovieLens dataset
- `/ContentBased`: Implementations of content-based filtering algorithms
- `/CollaborativeFiltering`: Implementations of collaborative filtering algorithms
- `/MatrixFactorization`: Implementations of matrix factorization techniques
- `/EvaluationFramework`: Custom framework for evaluating algorithm performance
- `/notebooks`: Jupyter notebooks for exploration and model development
- `/models`: Saved model files

## Acknowledgments
- MovieLens dataset provided by GroupLens Research at the University of Minnesota
- This project is for educational and research purposes only