import pandas as pd
import numpy as np
from Recommender.base import RecommendationModel
from Recommender.content_based import ContentBasedModel
from Recommender.collaborative_filtering import CollaborativeFilteringModel

class HybridRecommendationModel(RecommendationModel):
    """
    Hybrid recommendation model that combines content-based and collaborative filtering approaches.
    This demonstrates advanced inheritance by using composition rather than direct inheritance
    from multiple classes, which is generally a better practice in Python.
    """
    def __init__(self, movies_df, ratings_df=None, content_weight=0.5, collab_weight=0.5):
        self.movies_df = movies_df
        self.content_model = ContentBasedModel(movies_df)
        self.collaborative_model = CollaborativeFilteringModel(movies_df, ratings_df)

        # Weights for combining recommendations
        self.content_weight = content_weight
        self.collab_weight = collab_weight

        # Normalize weights
        total_weight = self.content_weight + self.collab_weight
        self.content_weight /= total_weight
        self.collab_weight /= total_weight

    def recommend(self, preferences: dict, n: int = 5):
        """
        Generate recommendations using a hybrid approach.

        Args:
            preferences: Dictionary containing user preferences
            n: Number of recommendations to return

        Returns:
            DataFrame of recommended movies
        """
        # Get recommendations from each model
        content_recs = self.content_model.recommend(preferences, n=n*2)
        collab_recs = self.collaborative_model.recommend(preferences, n=n*2)

        # Combine recommendations with weights
        combined_recs = self._combine_recommendations(content_recs, collab_recs)

        # Apply diversity enhancement
        diverse_recs = self._enhance_diversity(combined_recs, n)

        return diverse_recs.head(n)

    def _combine_recommendations(self, content_recs, collab_recs):
        """
        Combine recommendations from content-based and collaborative filtering models.

        Args:
            content_recs: Recommendations from content-based model
            collab_recs: Recommendations from collaborative filtering model

        Returns:
            Combined recommendations DataFrame
        """
        # Normalize scores within each recommendation set
        if not content_recs.empty:
            content_recs['normalized_score'] = content_recs['vote_average'] / content_recs['vote_average'].max()

        if not collab_recs.empty:
            collab_recs['normalized_score'] = collab_recs['vote_average'] / collab_recs['vote_average'].max()

        # Combine recommendations
        all_recs = pd.concat([content_recs, collab_recs]).drop_duplicates(subset=['id'])

        # Ensure normalized_score column exists in all_recs
        if 'normalized_score' not in all_recs.columns:
            all_recs['normalized_score'] = all_recs['vote_average'] / all_recs['vote_average'].max() if not all_recs.empty else 0

        # Calculate hybrid score
        all_recs['hybrid_score'] = 0.0

        # Add content-based score
        content_ids = set(content_recs['id']) if not content_recs.empty else set()
        if content_ids:
            all_recs.loc[all_recs['id'].isin(content_ids), 'hybrid_score'] += \
                all_recs.loc[all_recs['id'].isin(content_ids), 'normalized_score'] * self.content_weight

        # Add collaborative filtering score
        collab_ids = set(collab_recs['id']) if not collab_recs.empty else set()
        if collab_ids:
            all_recs.loc[all_recs['id'].isin(collab_ids), 'hybrid_score'] += \
                all_recs.loc[all_recs['id'].isin(collab_ids), 'normalized_score'] * self.collab_weight

        # Sort by hybrid score
        return all_recs.sort_values('hybrid_score', ascending=False)

    def _enhance_diversity(self, recommendations, n):
        """
        Enhance diversity in recommendations using a greedy approach.

        Args:
            recommendations: DataFrame of recommendations
            n: Number of recommendations to return

        Returns:
            DataFrame with diverse recommendations
        """
        if len(recommendations) <= n:
            return recommendations

        # Start with the highest scored item
        diverse_indices = [0]
        candidate_indices = list(range(1, len(recommendations)))

        # Greedy algorithm to maximize diversity
        while len(diverse_indices) < n and candidate_indices:
            max_diversity_idx = None
            max_diversity = -1

            for idx in candidate_indices:
                # Calculate diversity as the sum of genre differences
                diversity = self._calculate_diversity(recommendations.iloc[idx], 
                                                     recommendations.iloc[diverse_indices])

                if diversity > max_diversity:
                    max_diversity = diversity
                    max_diversity_idx = idx

            if max_diversity_idx is not None:
                diverse_indices.append(max_diversity_idx)
                candidate_indices.remove(max_diversity_idx)
            else:
                break

        return recommendations.iloc[diverse_indices]

    def _calculate_diversity(self, movie, selected_movies):
        """
        Calculate diversity between a movie and a set of already selected movies.

        Args:
            movie: A movie to calculate diversity for
            selected_movies: DataFrame of already selected movies

        Returns:
            Diversity score
        """
        # Calculate genre diversity
        movie_genres = set(movie['genres'])

        # Sum of Jaccard distances (1 - intersection/union)
        diversity = 0
        for _, selected_movie in selected_movies.iterrows():
            selected_genres = set(selected_movie['genres'])

            if not movie_genres and not selected_genres:
                continue

            intersection = len(movie_genres.intersection(selected_genres))
            union = len(movie_genres.union(selected_genres))

            # Jaccard distance
            if union > 0:
                diversity += 1 - (intersection / union)

        return diversity


class MatrixFactorizationModel(RecommendationModel):
    """
    Matrix Factorization recommendation model using Singular Value Decomposition (SVD).
    This is a technique commonly used by Netflix and other streaming platforms.
    """
    def __init__(self, movies_df, ratings_df=None, n_factors=50):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.n_factors = n_factors

        # If no ratings are provided, create dummy ratings
        if self.ratings_df is None:
            self.ratings_df = self._create_dummy_ratings()

        # Create user-item matrix
        self.user_item_matrix = self._create_user_item_matrix()

        # Perform matrix factorization
        self.U, self.sigma, self.Vt = self._perform_svd()

    def _create_dummy_ratings(self):
        """Create dummy ratings data for demonstration purposes."""
        # Similar to the method in CollaborativeFilteringModel
        n_users = 100
        n_movies = len(self.movies_df)

        ratings = []
        for user_id in range(1, n_users + 1):
            n_ratings = np.random.randint(1, 11)
            movie_indices = np.random.choice(n_movies, n_ratings, replace=False)

            for idx in movie_indices:
                movie_id = self.movies_df.iloc[idx].get('id', idx)
                rating = np.random.uniform(1, 5)
                ratings.append({
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'rating': rating
                })

        return pd.DataFrame(ratings)

    def _create_user_item_matrix(self):
        """Create a user-item matrix from ratings data."""
        return self.ratings_df.pivot_table(
            index='user_id', 
            columns='movie_id', 
            values='rating',
            fill_value=0
        )

    def _perform_svd(self):
        """Perform Singular Value Decomposition on the user-item matrix."""
        # Fill missing values with zeros
        matrix = self.user_item_matrix.fillna(0).values

        # Perform SVD
        U, sigma, Vt = np.linalg.svd(matrix, full_matrices=False)

        # Truncate to n_factors
        U = U[:, :self.n_factors]
        sigma = sigma[:self.n_factors]
        Vt = Vt[:self.n_factors, :]

        return U, sigma, Vt

    def _predict_ratings(self, user_idx):
        """Predict ratings for a user based on the factorized matrices."""
        user_factors = self.U[user_idx, :]
        predicted_ratings = np.dot(np.dot(user_factors, np.diag(self.sigma)), self.Vt)
        return predicted_ratings

    def recommend(self, preferences: dict, n: int = 5):
        """
        Recommend movies using matrix factorization.

        Args:
            preferences: Dictionary containing user preferences
            n: Number of recommendations to return

        Returns:
            DataFrame of recommended movies
        """
        user_id = preferences.get('user_id', None)
        min_rating = preferences.get('min_rating', 7.0)
        release_year_range = preferences.get('release_year', (1900, 2025))

        # If no user_id provided or user not in matrix, use a random user
        if user_id is None or user_id not in self.user_item_matrix.index:
            user_id = np.random.choice(self.user_item_matrix.index)

        # Get user index
        user_idx = self.user_item_matrix.index.get_loc(user_id)

        # Predict ratings for this user
        predicted_ratings = self._predict_ratings(user_idx)

        # Create a Series with movie_ids as index and predicted ratings as values
        movie_ids = self.user_item_matrix.columns
        predicted_df = pd.Series(predicted_ratings, index=movie_ids)

        # Filter out movies the user has already rated
        user_rated_movies = self.user_item_matrix.loc[user_id]
        user_rated_movies = user_rated_movies[user_rated_movies > 0].index
        predicted_df = predicted_df.drop(user_rated_movies, errors='ignore')

        # Get top N movie IDs based on predicted ratings
        top_movie_ids = predicted_df.sort_values(ascending=False).head(n*2).index

        # Get movie details
        recommended_movies = self.movies_df[self.movies_df['id'].isin(top_movie_ids)]

        # Apply additional filters from preferences
        filtered = recommended_movies[
            (recommended_movies['vote_average'] >= min_rating) &
            (recommended_movies['release_date'].str[:4].astype(int).between(*release_year_range))
        ]

        # If preferred genres are specified, prioritize those
        if 'genres' in preferences and preferences['genres']:
            preferred_genres = preferences['genres']
            filtered['genre_match'] = filtered['genres'].apply(
                lambda g: sum(genre in g for genre in preferred_genres)
            )
            return filtered.sort_values(['genre_match', 'vote_average'], ascending=[False, False]).head(n)

        # Otherwise, just sort by rating
        return filtered.sort_values('vote_average', ascending=False).head(n)
