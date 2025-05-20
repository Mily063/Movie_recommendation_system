import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Recommender.content_based import ContentBasedModel
from Recommender.collaborative_filtering import CollaborativeFilteringModel
from Recommender.hybrid import HybridRecommendationModel
import pandas as pd
import pytest


@pytest.fixture
def movies_df():
    return pd.DataFrame(
        [
            {
                "id": 1,
                "title": "Action Movie",
                "genres": ["Action"],
                "vote_average": 8.0,
                "release_date": "2020-01-01",
            },
            {
                "id": 2,
                "title": "Comedy Movie",
                "genres": ["Comedy"],
                "vote_average": 7.5,
                "release_date": "2019-05-05",
            },
            {
                "id": 3,
                "title": "Drama Movie",
                "genres": ["Drama"],
                "vote_average": 6.5,
                "release_date": "2018-08-20",
            },
            {
                "id": 4,
                "title": "Action-Comedy",
                "genres": ["Action", "Comedy"],
                "vote_average": 7.2,
                "release_date": "2022-03-21",
            },
        ]
    )


# TEST JEDNOSTKOWY
def test_hybrid_model_empty_case():
    empty_df = pd.DataFrame(
        columns=["id", "title", "genres", "vote_average", "release_date", "rating"]
    )
    hybrid = HybridRecommendationModel(empty_df)
    prefs = {"genres": ["Action"], "min_rating": 7.0, "release_year": (2018, 2023)}
    recs = hybrid.recommend(prefs, n=2)
    # Sprawdzamy, czy rekomendacje są puste
    assert recs.empty


# TEST INTEGRACYJNY
def test_content_and_collaborative_integration(movies_df):
    preferences = {
        "genres": ["Action"],
        "min_rating": 7.0,
        "release_year": (2018, 2023),
    }

    cb_model = ContentBasedModel(movies_df)
    cf_model = CollaborativeFilteringModel(movies_df)
    cb_recommendations = cb_model.recommend(preferences, n=2)
    cf_recommendations = cf_model.recommend(preferences, n=2)

    cb_titles = set(cb_recommendations["title"])
    cf_titles = set(cf_recommendations["title"])
    assert not cb_recommendations.empty
    assert not cf_recommendations.empty
    # Sprawdzamy czy istnieje chociaż jeden tytuł wspólny (integracja logiki doboru filmów)
    assert len(cb_titles & cf_titles) >= 1


# TEST END-TO-END
def test_hybrid_model_end_to_end(movies_df):
    prefs = {"genres": ["Action"], "min_rating": 7.0, "release_year": (2018, 2023)}
    model = HybridRecommendationModel(movies_df)
    recommendations = model.recommend(prefs, n=2)

    # Oczekujemy, że zwróci rekomendacje zawierające filmy z 'Action' i odpowiednią oceną/wiekiem
    assert not recommendations.empty
    assert all("Action" in genre_list for genre_list in recommendations["genres"])
    assert all(rating >= 7.0 for rating in recommendations["vote_average"])
    assert all(
        int(row["release_date"][:4]) >= 2018 for _, row in recommendations.iterrows()
    )
