from Recommender.base import RecommendationModel


class ContentBasedModel(RecommendationModel):
    def __init__(self, movies_df):
        self.movies_df = movies_df

    def recommend(self, preferences: dict, n: int = 5):
        preferred_genres = preferences.get("genres", [])
        min_rating = preferences.get("min_rating", 7.0)
        release_year_range = preferences.get("release_year", (1900, 2025))

        # Filter by genres and ratings first
        filtered = self.movies_df[
            (
                self.movies_df["genres"].apply(
                    lambda g: any(genre in g for genre in preferred_genres)
                )
            )
            & (self.movies_df["vote_average"] >= min_rating)
        ]

        # Filter by release year
        filtered = filtered[
            filtered["release_date"].str[:4].str.match(r"\d{4}", na=False)
            & filtered["release_date"]
            .str[:4]
            .astype(float)
            .between(*release_year_range)
        ]

        filtered = filtered.drop_duplicates(subset=["id"])

        return filtered.sort_values("vote_average", ascending=False).head(n)
