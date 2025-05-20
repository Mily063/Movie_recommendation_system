import pandas as pd
import numpy as np


def calculate_movie_average_rating(ratings_df, movie_id):
    """
    Oblicza średnią ocenę dla danego filmu.

    Args:
        ratings_df: DataFrame z ocenami
        movie_id: ID filmu

    Returns:
        Średnia ocena filmu lub 0 jeśli nikt nie ocenił filmu
    """
    # TODO: Zaimplementuj funkcję obliczającą średnią ocenę filmu
    # 1. Wyfiltruj oceny dla danego filmu
    # 2. Sprawdź czy istnieją oceny tego filmu
    # 3. Jeśli nie ma ocen, zwróć 0
    # 4. W przeciwnym przypadku zwróć średnią ocenę
    pass


def calculate_user_average_rating(ratings_df, user_id):
    """
    Oblicza średnią ocenę wszystkich filmów dla danego użytkownika.

    Args:
        ratings_df: DataFrame z ocenami
        user_id: ID użytkownika

    Returns:
        Średnia ocena użytkownika lub 0 jeśli użytkownik nie ocenił żadnego filmu
    """
    # TODO: Zaimplementuj funkcję obliczającą średnią ocenę użytkownika
    # 1. Wyfiltruj oceny danego użytkownika
    # 2. Sprawdź czy użytkownik ocenił jakiekolwiek filmy
    # 3. Jeśli nie ocenił żadnego, zwróć 0
    # 4. W przeciwnym przypadku zwróć średnią ocenę
    pass


def get_common_ratings(ratings_df, user_id1, user_id2):
    """
    Funkcja pomocnicza zwracająca oceny wspólnych filmów.

    Args:
        ratings_df: DataFrame z ocenami
        user_id1: ID pierwszego użytkownika
        user_id2: ID drugiego użytkownika

    Returns:
        Dwie listy ocen wspólnych filmów
    """
    # Pobierz oceny obu użytkowników
    user1_ratings = ratings_df[ratings_df['user_id'] == user_id1]
    user2_ratings = ratings_df[ratings_df['user_id'] == user_id2]

    # Znajdź wspólne filmy
    common_movies = set(user1_ratings['movie_id']).intersection(set(user2_ratings['movie_id']))

    # TODO: Dokończ funkcję
    # 1. Utwórz dwie puste listy: ratings1 i ratings2
    # 2. Dla każdego filmu z common_movies, dodaj jego ocenę do odpowiedniej listy
    # 3. Zwróć obie listy
    pass


def predict_movie_rating(ratings_df, user_id, movie_id):
    """
    Przewiduje ocenę filmu dla użytkownika.

    Args:
        ratings_df: DataFrame z ocenami
        user_id: ID użytkownika
        movie_id: ID filmu

    Returns:
        Przewidywana ocena filmu
    """
    # Sprawdź, czy użytkownik już ocenił film
    user_rating = ratings_df[(ratings_df['user_id'] == user_id) &
                             (ratings_df['movie_id'] == movie_id)]

    if not user_rating.empty:
        return user_rating['rating'].iloc[0]

    # Znajdź użytkowników, którzy ocenili ten film
    users_rated_movie = ratings_df[
        (ratings_df['movie_id'] == movie_id) &
        (ratings_df['user_id'] != user_id)
        ]['user_id'].unique()

    if len(users_rated_movie) == 0:
        return calculate_movie_average_rating(ratings_df, movie_id)

    # Poniższy kod pomoże w obliczeniu podobieństwa:
    similarities = []
    for other_user in users_rated_movie:
        ratings1, ratings2 = get_common_ratings(ratings_df, user_id, other_user)

        if len(ratings1) > 0:
            # Oblicz podobieństwo (korelację Pearsona)
            similarity = np.corrcoef(ratings1, ratings2)[0, 1]

            # Obsłuż przypadek NaN
            if np.isnan(similarity):
                similarity = 0

            similarities.append((other_user, similarity))

    # TODO: Wybierz najbardziej podobnych użytkowników i oblicz przewidywaną ocenę
    # 1. Jeśli lista similarities jest pusta, zwróć średnią ocenę filmu
    # 2. Posortuj listę similarities malejąco według podobieństwa - możesz użyć sort()
    # 3. Wybierz 2 najbardziej podobnych użytkowników
    # 4. Oblicz średnią ważoną ich ocen (waga = podobieństwo)
    # 5. Zwróć przewidywaną ocenę
    pass


# Kod testujący
if __name__ == "__main__":
    # Przykładowe dane ocen
    ratings_data = {
        'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'movie_id': [101, 102, 103, 101, 103, 104, 101, 102, 104, 102, 103, 104],
        'rating': [5.0, 3.0, 4.5, 4.0, 3.5, 5.0, 2.5, 4.0, 4.5, 3.0, 4.0, 3.5]
    }
    ratings_df = pd.DataFrame(ratings_data)

    # Test 1: Obliczanie średniej oceny filmu
    # avg_rating = calculate_movie_average_rating(ratings_df, movie_id=104)
    # print(f"Średnia ocena filmu 104: {avg_rating:.2f}")
    # print(f"Oczekiwany wynik: 4.33")

    # Test 2: Obliczanie średniej oceny użytkownika
    # user_avg = calculate_user_average_rating(ratings_df, user_id=1)
    # print(f"Średnia ocena użytkownika 1: {user_avg:.2f}")
    # print(f"Oczekiwany wynik: 4.17")

    # Test 3: Przewidywanie oceny filmu
    # predicted_rating = predict_movie_rating(ratings_df, user_id=1, movie_id=104)
    # print(f"Przewidywana ocena filmu 104 przez użytkownika 1: {predicted_rating:.2f}")
    # print(f"Oczekiwany wynik: 4.25")