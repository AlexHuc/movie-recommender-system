import itertools

from surprise import accuracy
from collections import defaultdict

class RecommenderMetrics:
    """
    A suite of static methods for evaluating recommender system predictions.
    Includes error metrics and recommendation quality metrics.
    The metrics include:
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - GetTopN
        - HitRate
        - CumulativeHitRate (cHR)
        - RatingHitRate (rHR)
        - AverageReciprocalHitRank (ARHR)
        - UserCoverage
        - Diversity
        - Novelty
    """

    def MAE(predictions):
        """
        Compute Mean Absolute Error (MAE) for a set of predictions.

        Args:
            predictions (list of Prediction): List of Surprise Prediction objects.
        Returns:
            float: The average absolute difference between actual and estimated ratings.
        """
        return accuracy.mae(predictions, verbose=False)

    def RMSE(predictions):
        """
        Compute Root Mean Squared Error (RMSE) for a set of predictions.

        Args:
            predictions (list of Prediction): List of Surprise Prediction objects.
        Returns:
            float: The square root of the average squared difference between actual and estimated ratings.
        """
        return accuracy.rmse(predictions, verbose=False)

    def GetTopN(predictions, n=10, minimumRating=4.0):
        """
        Return the top-N highest estimated ratings for each user, filtered by a minimum rating threshold.

        Args:
            predictions (list of Prediction): List of Surprise Prediction objects.
            n (int): Number of top items to keep per user.
            minimumRating (float): Threshold for estimated rating to be considered.

        Returns:
            dict[int, list[tuple[int, float]]]: Mapping from userID to list of (movieID, estimatedRating),
                                                sorted descending by rating, length <= n.
        """
        # Temporarily store all high-rated predictions per user
        topN = defaultdict(list)
        # Iterate through all predictions
        for userID, movieID, actualRating, estimatedRating, _ in predictions:
            if (estimatedRating >= minimumRating):
                topN[int(userID)].append((int(movieID), estimatedRating))
        # Sort each user's list by estimated rating and trim to n
        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(userID)] = ratings[:n]

        return topN

    def HitRate(topNPredicted, leftOutPredictions):
        """
        Measure how often a left-out movie appears in the user's top-N recommendations.

        This is leave-one-out evaluation: for each hidden rating, check if it is
        in the predicted top-N list for that user.

        Args:
            topNPredicted (dict[int, list[tuple[int, float]]]): Top-N predictions per user.
            leftOutPredictions (list of tuples): Each tuple (userID, movieID, actual, estimated, _) representing hidden data.

        Returns:
            float: Hit rate = hits / total left-out ratings.
        """
        hits = 0
        total = 0
        # For each left-out rating
        for leftOut in leftOutPredictions:
            userID = leftOut[0]
            leftOutMovieID = leftOut[1]
            # Is it in the predicted top 10 for this user?
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == int(movieID)):
                    hit = True
                    break
            if (hit) :
                hits += 1

            total += 1

        # Compute overall precision
        return hits/total

    def CumulativeHitRate(topNPredicted, leftOutPredictions, ratingCutoff=0):
        """
        Hit rate only for left-out items with actual rating >= ratingCutoff.

        Args:
            topNPredicted (dict[int, list[tuple[int, float]]]): Top-N predictions per user.
            leftOutPredictions (list of tuples): (userID, movieID, actual, estimated, _).
            ratingCutoff (float): Minimum actual rating to include in evaluation.

        Returns:
            float: Hit rate over qualifying left-out ratings.
        """
        hits = 0
        total = 0

        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Only look at ability to recommend things the users actually liked...
            if (actualRating >= ratingCutoff):
                # Is it in the predicted top 10 for this user?
                hit = False
                for movieID, predictedRating in topNPredicted[int(userID)]:
                    if (int(leftOutMovieID) == movieID):
                        hit = True
                        break
                if (hit) :
                    hits += 1

                total += 1

        # Compute overall precision
        return hits/total

    def RatingHitRate(topNPredicted, leftOutPredictions):
        """
        Print hit rate for each actual rating value.

        Args:
            topNPredicted (dict[int, list[tuple[int, float]]]): Top-N per user.
            leftOutPredictions (list of tuples): (userID, movieID, actual, estimated, _).

        Returns:
            None: Prints rating vs. hit rate to stdout.
        """
        hits = defaultdict(float)
        total = defaultdict(float)

        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == movieID):
                    hit = True
                    break
            if (hit) :
                hits[actualRating] += 1

            total[actualRating] += 1

        # Compute overall precision
        for rating in sorted(hits.keys()):
            print (rating, hits[rating] / total[rating])

    def AverageReciprocalHitRank(topNPredicted, leftOutPredictions):
        """
        Compute the average reciprocal rank of left-out items in top-N lists.

        Args:
            topNPredicted (dict[int, list[tuple[int, float]]]): Top-N per user.
            leftOutPredictions (list of tuples): (userID, movieID, actual, estimated, _).

        Returns:
            float: Mean of reciprocal ranks (1/rank) over all left-out items.
        """
        summation = 0
        total = 0
        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
             # Find rank position in top-N
            hitRank = 0
            rank = 0
            for movieID, predictedRating in topNPredicted[int(userID)]:
                rank = rank + 1
                if (int(leftOutMovieID) == movieID):
                    hitRank = rank
                    break
            if (hitRank > 0) :
                summation += 1.0 / hitRank

            total += 1

        return summation / total

    # What percentage of users have at least one "good" recommendation
    def UserCoverage(topNPredicted, numUsers, ratingThreshold=0):
        """
        Percentage of users with at least one recommendation above a threshold.

        Args:
            topNPredicted (dict[int, list[tuple[int, float]]]): Top-N per user.
            numUsers (int): Total number of users in test set.
            ratingThreshold (float): Minimum predicted rating to count as a hit.

        Returns:
            float: Fraction of users with >=1 hit.
        """
        hits = 0
        for userID in topNPredicted.keys():
            hit = False
            for movieID, predictedRating in topNPredicted[userID]:
                if (predictedRating >= ratingThreshold):
                    hit = True
                    break
            if (hit):
                hits += 1

        return hits / numUsers

    def Diversity(topNPredicted, simsAlgo):
        """
        Measure recommendation diversity based on item-item similarity.

        Uses the similarity matrix from a trained Surprise similarity algorithm.

        Args:
            topNPredicted (dict[int, list[tuple[int, float]]]): Top-N per user.
            simsAlgo: Trained Surprise similarity algorithm with .compute_similarities().

        Returns:
            float: 1 minus average pairwise similarity of recommendations.
        """
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for userID in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[userID], 2)
            for pair in pairs:
                movie1 = pair[0][0]
                movie2 = pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(str(movie1))
                innerID2 = simsAlgo.trainset.to_inner_iid(str(movie2))
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                n += 1

        S = total / n
        return (1-S)

    def Novelty(topNPredicted, rankings):
        """
        Measure novelty as average popularity rank of recommended items.

        Lower average rank => more novel recommendations.

        Args:
            topNPredicted (dict[int, list[tuple[int, float]]]): Top-N per user.
            rankings (dict[int, int]): Precomputed movie popularity ranks.

        Returns:
            float: Mean rank of all recommended items.
        """
        n = 0
        total = 0
        for userID in topNPredicted.keys():
            for rating in topNPredicted[userID]:
                movieID = rating[0]
                rank = rankings[movieID]
                total += rank
                n += 1
        return total / n
