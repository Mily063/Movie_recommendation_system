from Recommender.base import RecommendationModel

class ContentBasedModel(RecommendationModel):
    def __init__(self, movies_df):
        self.movies_df = movies_df

    def recommend(self, preferences: dict, n: int = 5):
        preferred_genres = preferences.get('genres', [])
        min_rating = preferences.get('min_rating', 7.0)
        release_year_range = preferences.get('release_year', (1900, 2025))

        filtered = self.movies_df[
            (self.movies_df['genres'].apply(lambda g: any(genre in g for genre in preferred_genres))) &
            (self.movies_df['vote_average'] >= min_rating) &
            (self.movies_df['release_date'].str[:4].astype(int).between(*release_year_range))
            ]

        return filtered.sort_values('vote_average', ascending=False).head(n)