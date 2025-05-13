import requests
import pandas as pd

class MovieFetcher:
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_popular_movies(self,pages=1):
        movies = []
        genre_mapping = self.fetch_genre_mapping()  # Fetch genre mapping
        for page in range(1, pages + 20):
            url = f'https://api.themoviedb.org/3/movie/popular?api_key={self.api_key}&language=en-US&page={page}'
            response = requests.get(url).json()
            for movie in response.get('results', []):
                # Map genre IDs to names
                movie['genres'] = [genre_mapping.get(genre_id, "Unknown") for genre_id in movie.get('genre_ids', [])]
                movies.append(movie)
        return pd.DataFrame(movies)

    def fetch_genre_mapping(self):
        url = f'https://api.themoviedb.org/3/genre/movie/list?api_key={self.api_key}&language=en-US'
        response = requests.get(url).json()
        return {genre['id']: genre['name'] for genre in response.get('genres', [])}