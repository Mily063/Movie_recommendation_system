import pandas as pd
import numpy as np


def predict_movie_rating(ratings_df, user_id, movie_id, top_n_similar=2):
    """
    Przewiduje ocenę filmu dla użytkownika na podstawie ocen podobnych użytkowników.

    Args:
        ratings_df: DataFrame z kolumnami ['user_id', 'movie_id', 'rating']
        user_id: ID użytkownika, dla którego przewidujemy ocenę
        movie_id: ID filmu, dla którego przewidujemy ocenę
        top_n_similar: Liczba podobnych użytkowników do uwzględnienia

    Returns:
        Przewidywana ocena filmu
    """
    # Sprawdź, czy użytkownik nie ocenił już filmu
    if not ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['movie_id'] == movie_id)].empty:
        return ratings_df[(ratings_df['user_id'] == user_id) &
                          (ratings_df['movie_id'] == movie_id)]['rating'].iloc[0]

    # Znajdź użytkowników, którzy ocenili ten film
    users_who_rated = ratings_df[ratings_df['movie_id'] == movie_id]['user_id'].unique()

    # TODO: Uzupełnij kod poniżej, aby przewidzieć ocenę filmu
    # 1. Oblicz średnią ocenę filmu (jako wartość domyślną)
    # 2. Jeśli nie ma użytkowników, którzy ocenili film, zwróć 0
    # 3. Dla każdego użytkownika, który ocenił film, oblicz jego średnią ocenę wszystkich filmów
    # 4. Zwróć średnią ocenę filmu z top_n_similar użytkowników


# Kod testujący
if __name__ == "__main__":
    # Przykładowe dane ocen
    ratings_data = {
        'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'movie_id': [101, 102, 103, 101, 103, 104, 101, 102, 104, 102, 103, 104],
        'rating': [5.0, 3.0, 4.5, 4.0, 3.5, 5.0, 2.5, 4.0, 4.5, 3.0, 4.0, 3.5]
    }
    ratings_df = pd.DataFrame(ratings_data)

    # Test dla użytkownika 1 i filmu 104 (którego nie ocenił)
    predicted_rating = predict_movie_rating(ratings_df, user_id=1, movie_id=104)
    print(f"Przewidywana ocena filmu 104, przez użytkownika 1: {predicted_rating:.2f}")