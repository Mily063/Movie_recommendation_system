import pandas as pd


def find_popular_movies(ratings_df, min_ratings=2):
    """
    Funkcja znajdująca najpopularniejsze filmy na podstawie średnich ocen.

    Args:
        ratings_df: DataFrame z kolumnami ['user_id', 'movie_id', 'rating']
        min_ratings: Minimalna liczba ocen, aby film był uwzględniony

    Returns:
        DataFrame z najpopularniejszymi filmami, posortowany według średniej oceny
    """
    # TODO: Uzupełnij kod poniżej, aby znaleźć najpopularniejsze filmy
    # 1. Zlicz liczbę ocen dla każdego filmu
    # 2. Oblicz średnią ocenę dla każdego filmu
    # 3. Odfiltruj filmy z liczbą ocen mniejszą niż min_ratings
    # 4. Posortuj filmy malejąco według średniej oceny
    # 5. Zwróć posortowany DataFrame


# Kod testujący
if __name__ == "__main__":
    # Przykładowe dane ocen
    ratings_data = {
        'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5],
        'movie_id': [101, 102, 103, 101, 103, 104, 101, 102, 104, 102, 103, 104, 105],
        'rating': [5.0, 3.0, 4.5, 4.0, 3.5, 5.0, 2.5, 4.0, 4.5, 3.0, 4.0, 3.5, 5.0]
    }
    ratings_df = pd.DataFrame(ratings_data)

    # Dodaj informacje o filmach dla czytelności
    movies_data = {
        'movie_id': [101, 102, 103, 104, 105],
        'title': ['Film A', 'Film B', 'Film C', 'Film D', 'Film E']
    }
    movies_df = pd.DataFrame(movies_data)

    # Znajdź popularne filmy
    popular_movies = find_popular_movies(ratings_df, min_ratings=2)

    # Dodaj tytuły filmów
    popular_movies_with_titles = popular_movies.merge(movies_df, on='movie_id')

    print("Najpopularniejsze filmy:")
    print(popular_movies_with_titles[['title', 'mean_rating', 'rating_count']])