import pandas as pd
import numpy as np
from Recommender.base import RecommendationModel


class HybridRecommendationModel(RecommendationModel):
    def __init__(self, movies_df, content_weight=0.5, collab_weight=0.5):
        from Recommender.content_based import ContentBasedModel
        from Recommender.collaborative_filtering import CollaborativeFilteringModel

        self.movies_df = movies_df
        self.content_weight = content_weight
        self.collab_weight = collab_weight
        self.content_model = ContentBasedModel(movies_df)
        self.collaborative_model = CollaborativeFilteringModel(movies_df)

    def recommend(self, preferences: dict, n: int = 5):
        # Pobierz rekomendacje z dwóch modeli
        rec_content = self.content_model.recommend(preferences, n=20)
        rec_collab = self.collaborative_model.recommend(preferences, n=20)

        ##
        # if (rec_content is None or rec_content.empty) and (
        #     rec_collab is None or rec_collab.empty
        # ):
        #     # Zwróć pusty DataFrame z zachowaniem kolumn wejściowych
        #     return pd.DataFrame(columns=self.movies_df.columns)
        ##

        rec_content = rec_content.copy()
        rec_collab = rec_collab.copy()
        rec_content["hybrid_score"] = self.content_weight * (
            rec_content["vote_average"].rank(method="min", ascending=False)
            / rec_content.shape[0]
        )
        rec_collab["hybrid_score"] = self.collab_weight * (
            rec_collab["vote_average"].rank(method="min", ascending=False)
            / rec_collab.shape[0]
        )
        combined = pd.concat([rec_content, rec_collab])
        combined = (
            combined.groupby("id")
            .agg(
                {
                    "title": "first",
                    "genres": "first",
                    "vote_average": "mean",
                    "release_date": "first",
                    "hybrid_score": "sum",
                }
            )
            .reset_index()
        )

        # Filtrowanie końcowe wg preferencji
        min_rating = preferences.get("min_rating", 7.0)
        year_range = preferences.get("release_year", (1900, 2025))
        preferred_genres = preferences.get("genres", [])

        filtered = combined[
            (combined["vote_average"] >= min_rating)
            & (combined["release_date"].str[:4].astype(float).between(*year_range))
        ].copy()

        # Priorytetyzacja po preferred_genres
        if preferred_genres:
            filtered["genre_match"] = filtered["genres"].apply(
                lambda g: sum(genre in g for genre in preferred_genres)
            )
            filtered = filtered.sort_values(
                ["genre_match", "hybrid_score", "vote_average"],
                ascending=[False, False, False],
            )
        else:
            filtered = filtered.sort_values(
                ["hybrid_score", "vote_average"], ascending=[False, False]
            )
        filtered = filtered.drop_duplicates(subset=["id"])
        return filtered.head(n)

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
            content_recs["normalized_score"] = (
                content_recs["vote_average"] / content_recs["vote_average"].max()
            )

        # Combine recommendations
        all_recs = pd.concat([content_recs, collab_recs]).drop_duplicates(subset=["id"])

        # Ensure normalized_score column exists in all_recs
        if "normalized_score" not in all_recs.columns:
            all_recs["normalized_score"] = (
                all_recs["vote_average"] / all_recs["vote_average"].max()
                if not all_recs.empty
                else 0
            )

        # Calculate hybrid score
        all_recs["hybrid_score"] = 0.0

        # Add content-based score
        content_ids = set(content_recs["id"]) if not content_recs.empty else set()
        if content_ids:
            all_recs.loc[all_recs["id"].isin(content_ids), "hybrid_score"] += (
                all_recs.loc[all_recs["id"].isin(content_ids), "normalized_score"]
                * self.content_weight
            )

        # Add collaborative filtering score
        collab_ids = set(collab_recs["id"]) if not collab_recs.empty else set()
        if collab_ids:
            all_recs.loc[all_recs["id"].isin(collab_ids), "hybrid_score"] += (
                all_recs.loc[all_recs["id"].isin(collab_ids), "normalized_score"]
                * self.collab_weight
            )

        # Sort by hybrid score
        return all_recs.sort_values("hybrid_score", ascending=False)

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
                diversity = self._calculate_diversity(
                    recommendations.iloc[idx], recommendations.iloc[diverse_indices]
                )

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
        movie_genres = set(movie["genres"])

        # Sum of Jaccard distances (1 - intersection/union)
        diversity = 0
        for _, selected_movie in selected_movies.iterrows():
            selected_genres = set(selected_movie["genres"])

            if not movie_genres and not selected_genres:
                continue

            intersection = len(movie_genres.intersection(selected_genres))
            union = len(movie_genres.union(selected_genres))

            # Jaccard distance
            if union > 0:
                diversity += 1 - (intersection / union)

        return diversity


class MatrixFactorizationModel(RecommendationModel):
    def __init__(self, movies_df, n_factors=50, ratings_df=None):
        self.movies_df = movies_df
        self.n_factors = n_factors
        self.ratings_df = (
            ratings_df if ratings_df is not None else self._create_dummy_ratings()
        )
        self.user_item_matrix = self._create_user_item_matrix()
        self.U, self.sigma, self.Vt = self._perform_svd()

    def _create_dummy_ratings(self):
        n_users = 100
        n_movies = len(self.movies_df)
        ratings = []
        for user_id in range(1, n_users + 1):
            n_ratings = np.random.randint(5, 15)
            movie_indices = np.random.choice(n_movies, n_ratings, replace=False)
            for idx in movie_indices:
                movie_id = self.movies_df.iloc[idx].get("id", idx)
                rating = np.random.uniform(1, 5)
                ratings.append(
                    {"user_id": user_id, "movie_id": movie_id, "rating": rating}
                )
        return pd.DataFrame(ratings)

    def _create_user_item_matrix(self):
        return self.ratings_df.pivot_table(
            index="user_id", columns="movie_id", values="rating", fill_value=0
        )

    def _perform_svd(self):
        from scipy.sparse.linalg import svds

        mat = self.user_item_matrix.values
        U, sigma, Vt = svds(mat, k=min(self.n_factors, min(mat.shape) - 1))
        return U, sigma, Vt

    def recommend(self, preferences: dict, n: int = 5):
        user_id = preferences.get("user_id", None)
        min_rating = preferences.get("min_rating", 7.0)
        release_year = preferences.get("release_year", (1900, 2025))
        preferred_genres = preferences.get("genres", [])

        # Dobór użytkownika
        if user_id is None or user_id not in self.user_item_matrix.index:
            user_id = np.random.choice(self.user_item_matrix.index)
        user_idx = self.user_item_matrix.index.get_loc(user_id)

        # Przewidziane oceny
        user_ratings_pred = np.dot(
            self.U[user_idx], np.dot(np.diag(self.sigma), self.Vt)
        )
        all_movie_ids = self.user_item_matrix.columns

        # Wykluczaj już ocenione
        already_rated = set(
            self.user_item_matrix.loc[user_id][
                self.user_item_matrix.loc[user_id] > 0
            ].index
        )
        recommend_ids = [
            mid
            for i, mid in enumerate(all_movie_ids)
            if mid not in already_rated and user_ratings_pred[i] > 3.0
        ]

        rec_movies = self.movies_df[self.movies_df["id"].isin(recommend_ids)].copy()
        rec_movies["mf_score"] = rec_movies["id"].map(
            dict(zip(all_movie_ids, user_ratings_pred))
        )

        # Filtracja końcowa
        filtered = rec_movies[
            (rec_movies["vote_average"] >= min_rating)
            & (rec_movies["release_date"].str[:4].astype(float).between(*release_year))
        ].copy()

        if preferred_genres:
            filtered["genre_match"] = filtered["genres"].apply(
                lambda g: sum(genre in g for genre in preferred_genres)
            )
            filtered = filtered.sort_values(
                ["genre_match", "mf_score", "vote_average"],
                ascending=[False, False, False],
            )
        else:
            filtered = filtered.sort_values(
                ["mf_score", "vote_average"], ascending=[False, False]
            )
        filtered = filtered.drop_duplicates(subset=["id"])
        return filtered.head(n)

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
        user_id = preferences.get("user_id", None)
        min_rating = preferences.get("min_rating", 7.0)
        release_year_range = preferences.get("release_year", (1900, 2025))

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
        predicted_df = predicted_df.drop(user_rated_movies, errors="ignore")

        # Get top N movie IDs based on predicted ratings
        top_movie_ids = predicted_df.sort_values(ascending=False).head(n * 2).index

        # Get movie details
        recommended_movies = self.movies_df[self.movies_df["id"].isin(top_movie_ids)]

        # Apply additional filters from preferences
        filtered = recommended_movies[
            (recommended_movies["vote_average"] >= min_rating)
            & (
                recommended_movies["release_date"]
                .str[:4]
                .astype(int)
                .between(*release_year_range)
            )
        ]

        # If preferred genres are specified, prioritize those
        if "genres" in preferences and preferences["genres"]:
            preferred_genres = preferences["genres"]
            filtered.loc[:, "genre_match"] = filtered["genres"].apply(
                lambda g: sum(genre in g for genre in preferred_genres)
            )
            return filtered.sort_values(
                ["genre_match", "vote_average"], ascending=[False, False]
            ).head(n)

        return filtered.sort_values("vote_average", ascending=False).head(n)
