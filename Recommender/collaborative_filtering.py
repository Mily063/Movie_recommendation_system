import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from Recommender.base import RecommendationModel

class CollaborativeFilteringModel(RecommendationModel):
    """
    Collaborative Filtering recommendation model that uses user-item interactions
    to make recommendations based on similar users' preferences.
    """
    def __init__(self, movies_df, ratings_df=None):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        
        # If no ratings are provided, create a dummy ratings dataframe
        if self.ratings_df is None:
            self.ratings_df = self._create_dummy_ratings()
        
        # Create user-item matrix
        self.user_item_matrix = self._create_user_item_matrix()
        
        # Compute similarity matrix
        self.similarity_matrix = self._compute_similarity_matrix()
    
    def _create_dummy_ratings(self):
        """Create dummy ratings data for demonstration purposes."""
        # In a real system, this would be actual user ratings
        # For now, we'll create random ratings for demonstration
        n_users = 100
        n_movies = len(self.movies_df)
        
        # Create sparse ratings (most users rate only a few movies)
        ratings = []
        for user_id in range(1, n_users + 1):
            # Each user rates 1-10 random movies
            n_ratings = np.random.randint(1, 11)
            movie_indices = np.random.choice(n_movies, n_ratings, replace=False)
            
            for idx in movie_indices:
                movie_id = self.movies_df.iloc[idx].get('id', idx)
                rating = np.random.uniform(1, 5)  # Random rating between 1 and 5
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
    
    def _compute_similarity_matrix(self):
        """Compute user-user similarity matrix using cosine similarity."""
        return cosine_similarity(self.user_item_matrix)
    
    def _get_similar_users(self, user_id, n=10):
        """Get the most similar users to the given user."""
        if user_id not in self.user_item_matrix.index:
            # If user not in matrix, return random users
            return np.random.choice(self.user_item_matrix.index, n, replace=False)
        
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        similar_users_idx = np.argsort(self.similarity_matrix[user_idx])[::-1][1:n+1]
        return self.user_item_matrix.index[similar_users_idx]
    
    def recommend(self, preferences: dict, n: int = 5):
        """
        Recommend movies based on collaborative filtering.
        
        Args:
            preferences: Dictionary containing user preferences
                - user_id: ID of the user (if available)
                - genres: List of preferred genres
                - min_rating: Minimum acceptable rating
                - release_year: Tuple of (min_year, max_year)
            n: Number of recommendations to return
            
        Returns:
            DataFrame of recommended movies
        """
        user_id = preferences.get('user_id', None)
        min_rating = preferences.get('min_rating', 7.0)
        release_year_range = preferences.get('release_year', (1900, 2025))
        
        # If no user_id provided, use a random user
        if user_id is None or user_id not in self.user_item_matrix.index:
            user_id = np.random.choice(self.user_item_matrix.index)
        
        # Get similar users
        similar_users = self._get_similar_users(user_id)
        
        # Get movies rated highly by similar users
        similar_users_ratings = self.user_item_matrix.loc[similar_users]
        mean_ratings = similar_users_ratings.mean()
        
        # Filter movies with sufficient ratings
        recommended_movie_ids = mean_ratings[mean_ratings > 3.5].index.tolist()
        
        # Get movie details and apply additional filters
        recommended_movies = self.movies_df[self.movies_df['id'].isin(recommended_movie_ids)]
        
        # Apply additional filters from preferences
        filtered = recommended_movies[
            (recommended_movies['vote_average'] >= min_rating) &
            (recommended_movies['release_date'].str[:4].astype(int).between(*release_year_range))
        ]
        
        # If preferred genres are specified, prioritize those
        if 'genres' in preferences and preferences['genres']:
            preferred_genres = preferences['genres']
            # Add a score column based on genre match
            filtered['genre_match'] = filtered['genres'].apply(
                lambda g: sum(genre in g for genre in preferred_genres)
            )
            # Sort by genre match and then by rating
            return filtered.sort_values(['genre_match', 'vote_average'], ascending=[False, False]).head(n)
        
        # Otherwise, just sort by rating
        return filtered.sort_values('vote_average', ascending=False).head(n)