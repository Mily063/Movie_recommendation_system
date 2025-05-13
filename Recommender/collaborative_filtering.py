import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from Recommender.base import RecommendationModel

class CollaborativeFilteringModel(RecommendationModel):
    """
    Collaborative Filtering recommendation model z dummy ratings (sztuczne dane),
    rekomenduje filmy na podstawie podobieństw użytkowników.
    """
    def __init__(self, movies_df, ratings_df=None):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        if self.ratings_df is None:
            self.ratings_df = self._create_dummy_ratings()
        self.user_item_matrix = self._create_user_item_matrix()
        self.similarity_matrix = self._compute_similarity_matrix()

    def _create_dummy_ratings(self):
        n_users = 100
        n_movies = len(self.movies_df)
        ratings = []
        for user_id in range(1, n_users + 1):
            n_ratings = np.random.randint(5, 16)
            movie_indices = np.random.choice(n_movies, n_ratings, replace=False)
            for idx in movie_indices:
                # id musi istnieć dla filmu! Jeśli nie, weź index
                movie_id = self.movies_df.iloc[idx]['id'] if 'id' in self.movies_df.columns else idx
                rating = np.random.uniform(1, 5)
                ratings.append({'user_id': user_id, 'movie_id': movie_id, 'rating': rating})
        return pd.DataFrame(ratings)

    def _create_user_item_matrix(self):
        return self.ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )

    def _compute_similarity_matrix(self):
        return cosine_similarity(self.user_item_matrix)

    def _get_similar_users(self, user_id, n=10):
        if user_id not in self.user_item_matrix.index:
            return np.random.choice(self.user_item_matrix.index, n, replace=False)
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        similarity_scores = self.similarity_matrix[user_idx]
        top_30_idx = np.argsort(similarity_scores)[::-1][1:31]  # 30 najbardziej podobnych
        best_candidates = self.user_item_matrix.index[top_30_idx]
        # losowo wybierz n z tych 30
        return np.random.choice(best_candidates, n, replace=False)

    def regenerate_dummy_ratings(self):
        self.ratings_df = self._create_dummy_ratings()
        self.user_item_matrix = self._create_user_item_matrix()
        self.similarity_matrix = self._compute_similarity_matrix()

    def recommend(self, preferences: dict, n: int = 5):
        # 1. Za każdym razem generuj nową macierz ocen — gwarancja unikalnych wyników
        self.regenerate_dummy_ratings()

        # 2. Pobierz preferencje użytkownika (lub ustaw domyślne)
        user_id = preferences.get('user_id', None)
        preferred_genres = preferences.get('genres', [])
        min_rating = preferences.get('min_rating', 1.0)
        release_year_range = preferences.get('release_year', (1900, 2025))

        # 3. Losowy user_id jeśli nie podano lub nie istnieje
        if user_id is None or user_id not in self.user_item_matrix.index:
            user_id = np.random.choice(self.user_item_matrix.index)

        # 4. Dobierz losowych sąsiadów spośród najbardziej podobnych (np. 10 z 30)
        similar_users = self._get_similar_users(user_id, n=10)
        similar_users_ratings = self.user_item_matrix.loc[similar_users]
        mean_ratings = similar_users_ratings.mean()

        # 5. Usuń filmy już ocenione przez użytkownika
        user_rated = set(self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index)
        candidate_ids = [mid for mid in mean_ratings.index if mid not in user_rated and mean_ratings[mid] > 0]

        # 6. Wstępnie wybierz tylko te filmy, które spełniają filtry (możesz wprowadzić dalsze ograniczenia niżej)
        candidates_df = self.movies_df[self.movies_df['id'].isin(candidate_ids)].copy()

        # 6. Filtrowanie po roku i ocenach
        if 'release_date' in candidates_df.columns:
            release_years = candidates_df['release_date'].astype(str).str[:4]
            mask_year = release_years.str.match(r'\d{4}', na=False)
            years_as_numeric = pd.to_numeric(release_years, errors='coerce')
            mask_range = years_as_numeric.between(*release_year_range)
            candidates_df = candidates_df[mask_year & mask_range]
        if 'vote_average' in candidates_df.columns:
            candidates_df = candidates_df[candidates_df['vote_average'] >= min_rating]

        # 7. Filtrowanie po gatunku
        if preferred_genres and 'genres' in candidates_df.columns:
            candidates_df = candidates_df[candidates_df['genres'].apply(lambda genres: any(g in genres for g in preferred_genres))]

        # --- TU ZABEZPIECZENIE:
        if candidates_df.empty or 'id' not in candidates_df.columns:
            fallback = self.movies_df.copy()
            # Stosuj te same filtry!
            if preferred_genres and 'genres' in fallback.columns:
                fallback = fallback[fallback['genres'].apply(lambda genres: any(g in genres for g in preferred_genres))]
            if 'release_date' in fallback.columns:
                release_years = fallback['release_date'].astype(str).str[:4]
                mask_year = release_years.str.match(r'\d{4}', na=False)
                years_as_numeric = pd.to_numeric(release_years, errors='coerce')
                mask_range = years_as_numeric.between(*release_year_range)
                fallback = fallback[mask_year & mask_range]
            if 'vote_average' in fallback.columns:
                fallback = fallback[fallback['vote_average'] >= min_rating]
            fallback['rnd'] = np.random.uniform(0, 0.01, size=len(fallback))
            return fallback.sort_values('rnd', ascending=False).head(n)
        # ----

        candidates_df['collab_score'] = candidates_df['id'].map(mean_ratings)
        # 10. Wprowadź delikatną losowość — przy tych samych danych wyniki się mogą zmieniać
        candidates_df['rnd'] = np.random.uniform(0, 0.01, size=len(candidates_df))

        # 11. Sortuj: najpierw collaborative score, potem losowość
        candidates_df = candidates_df.sort_values(['collab_score', 'rnd'], ascending=False)

        # 12. Wybierz top N wyników
        result = candidates_df.drop_duplicates(subset=['id']).head(n)
        if not result.empty:
            return result

        # 13. Fallback — jeżeli nic nie znaleziono, wybierz z całej bazy, ale też stosuj filtry
        fallback = self.movies_df[~self.movies_df['id'].isin(user_rated)].copy()
        if preferred_genres and 'genres' in fallback.columns:
            fallback = fallback[fallback['genres'].apply(lambda genres: any(g in genres for g in preferred_genres))]
        if 'release_date' in fallback.columns:
            release_years = fallback['release_date'].astype(str).str[:4]
            mask_year = release_years.str.match(r'\d{4}', na=False)
            mask_range = release_years.astype(float).between(*release_year_range)
            fallback = fallback[mask_year & mask_range]
        if 'vote_average' in fallback.columns:
            fallback = fallback[fallback['vote_average'] >= min_rating]
        fallback['rnd'] = np.random.uniform(0, 0.01, size=len(fallback))
        fallback = fallback.sort_values('rnd', ascending=False)
        return fallback.head(n)