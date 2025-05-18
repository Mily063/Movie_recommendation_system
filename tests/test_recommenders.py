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


def test_content_based_recommend_filtering(movies_df):
    model = ContentBasedModel(movies_df)
    prefs = {"genres": ["Action"], "min_rating": 7.0, "release_year": (2019, 2023)}
    recs = model.recommend(prefs, n=2)
    assert not recs.empty
    assert all("Action" in g for gs in recs["genres"] for g in [gs])


def test_content_based_return_limit(movies_df):
    model = ContentBasedModel(movies_df)
    prefs = {"genres": ["Comedy"], "min_rating": 5.0, "release_year": (2018, 2025)}
    recs = model.recommend(prefs, n=1)
    assert len(recs) == 1


def test_collaborative_filtering_basic(movies_df):
    model = CollaborativeFilteringModel(movies_df)
    prefs = {
        "user_id": 1,
        "genres": [],
        "min_rating": 5.0,
        "release_year": (2018, 2022),
    }
    recs = model.recommend(prefs, n=2)
    assert isinstance(recs, pd.DataFrame)
    assert list(recs.columns).count("title") > 0


def test_hybrid_model_combines_both(movies_df):
    hybrid = HybridRecommendationModel(movies_df)
    prefs = {"genres": ["Action"], "min_rating": 7.0, "release_year": (2018, 2023)}
    recs = hybrid.recommend(prefs, n=2)
    assert not recs.empty
    assert "title" in recs.columns


def test_hybrid_model_empty_case():
    empty_df = pd.DataFrame(
        columns=["id", "title", "genres", "vote_average", "release_date", "rating"]
    )
    hybrid = HybridRecommendationModel(empty_df)
    prefs = {"genres": ["Action"], "min_rating": 7.0, "release_year": (2018, 2023)}
    recs = hybrid.recommend(prefs, n=2)
    assert recs.empty
