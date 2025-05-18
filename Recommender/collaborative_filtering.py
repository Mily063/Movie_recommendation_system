import numpy as np
import pandas as pd
from Recommender.base import RecommendationModel


class CollaborativeFilteringModel(RecommendationModel):
    """
    Collaborative Filtering recommendation model z 3 predefiniowanymi użytkownikami,
    którzy mają konkretne, niezmienne preferencje.
    """

    def __init__(self, movies_df, ratings_df=None):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        if self.ratings_df is None:
            self.ratings_df = self._create_dummy_ratings()
        self.user_item_matrix = self._create_user_item_matrix()

    def _create_dummy_ratings(self):
        """
        Tworzenie ocen na podstawie trzech użytkowników o zdefiniowanych preferencjach.
        Przykładowo:
        - user_id=1 lubi tylko filmy z pierwszym gatunkiem,
        - user_id=2 lubi wysokie oceny z ostatnich lat,
        - user_id=3 lubi wszystko po równo.
        """
        ratings = []
        users = [1, 2, 3]
        for _, row in self.movies_df.iterrows():
            gid = row.get("id", row.name)
            genres = row["genres"] if "genres" in row else []
            year = (
                int(row["release_date"][:4])
                if "release_date" in row and str(row["release_date"])[:4].isdigit()
                else 2000
            )
            vote = float(row.get("vote_average", 3))

            # user 1 lubi konkretny gatunek, np. 'Action'
            ratings.append(
                {
                    "user_id": 1,
                    "movie_id": gid,
                    "rating": 5.0 if "Action" in str(genres) else 2.0,
                }
            )

            # user 2 lubi nowe i dobrze oceniane
            ratings.append(
                {
                    "user_id": 2,
                    "movie_id": gid,
                    "rating": 5.0 if vote > 7 and year > 2015 else 2.0,
                }
            )

            # user 3 losowe
            ratings.append({"user_id": 3, "movie_id": gid, "rating": 3.0})

        return pd.DataFrame(ratings)

    def _create_user_item_matrix(self):
        required_cols = {"user_id", "movie_id", "rating"}
        if self.ratings_df.empty or not required_cols.issubset(self.ratings_df.columns):
            return pd.DataFrame()
        return self.ratings_df.pivot_table(
            index="user_id", columns="movie_id", values="rating", fill_value=0
        )

    def _get_similar_users(self, user_id, n=10):
        other_users = [uid for uid in self.user_item_matrix.index if uid != user_id]
        return other_users[:n]

    def recommend(self, preferences: dict, n: int = 5):
        if self.user_item_matrix.empty:
            return pd.DataFrame()
        user_id = preferences.get("user_id", None)
        preferred_genres = preferences.get("genres", [])
        min_rating = preferences.get("min_rating", 1.0)
        release_year_range = preferences.get("release_year", (1900, 2025))

        # Dobieramy deteministycznie usera
        if user_id is None or user_id not in self.user_item_matrix.index:
            user_id = self.user_item_matrix.index[0]

        similar_users = self._get_similar_users(user_id, n=10)
        if not similar_users:
            return pd.DataFrame()  # Brak innych użytkowników

        similar_users_ratings = self.user_item_matrix.loc[similar_users]
        mean_ratings = similar_users_ratings.mean()

        user_rated = set(
            self.user_item_matrix.loc[user_id][
                self.user_item_matrix.loc[user_id] > 0
            ].index
        )
        candidate_ids = [
            mid
            for mid in mean_ratings.index
            if mid not in user_rated and mean_ratings[mid] > 0
        ]

        candidates_df = self.movies_df[self.movies_df["id"].isin(candidate_ids)].copy()

        if "release_date" in candidates_df.columns:
            release_years = candidates_df["release_date"].astype(str).str[:4]
            mask_year = release_years.str.match(r"\d{4}", na=False)
            years_as_numeric = pd.to_numeric(release_years, errors="coerce")
            mask_range = years_as_numeric.between(*release_year_range)
            candidates_df = candidates_df[mask_year & mask_range]
        if "vote_average" in candidates_df.columns:
            candidates_df = candidates_df[candidates_df["vote_average"] >= min_rating]
        if preferred_genres and "genres" in candidates_df.columns:
            candidates_df = candidates_df[
                candidates_df["genres"].apply(
                    lambda genres: any(g in genres for g in preferred_genres)
                )
            ]

        if candidates_df.empty or "id" not in candidates_df.columns:
            fallback = self.movies_df.copy()
            # Filtrowanie jak wyżej:
            if "release_date" in fallback.columns:
                release_years = fallback["release_date"].astype(str).str[:4]
                mask_year = release_years.str.match(r"\d{4}", na=False)
                years_as_numeric = pd.to_numeric(release_years, errors="coerce")
                mask_range = years_as_numeric.between(*release_year_range)
                fallback = fallback[mask_year & mask_range]
            if "vote_average" in fallback.columns:
                fallback = fallback[fallback["vote_average"] >= min_rating]
            if preferred_genres and "genres" in fallback.columns:
                fallback = fallback[
                    fallback["genres"].apply(
                        lambda genres: any(g in genres for g in preferred_genres)
                    )
                ]
            if fallback.empty or "id" not in fallback.columns:
                return pd.DataFrame()
            return fallback.sort_values("id").head(n)

        candidates_df["collab_score"] = candidates_df["id"].map(mean_ratings)
        candidates_df = candidates_df.sort_values(
            ["collab_score", "id"], ascending=[False, True]
        )
        result = candidates_df.drop_duplicates(subset=["id"]).head(n)
        if not result.empty:
            return result

        # Fallback na końcu
        fallback = self.movies_df.copy()
        if "id" in fallback.columns:
            return fallback.sort_values("id").head(n)
        return pd.DataFrame()
