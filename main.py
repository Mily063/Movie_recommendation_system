import os
import pandas as pd

import streamlit as st
from Recommender.content_based import ContentBasedModel
from Recommender.collaborative_filtering import CollaborativeFilteringModel
from Recommender.hybrid import HybridRecommendationModel, MatrixFactorizationModel
from Recommender.fetcher import MovieFetcher
from UI.interface import get_user_preferences

API_KEY = os.getenv("TMDB_API_KEY")

if not API_KEY:
    raise ValueError("API key not found. Please set the TMDB_API_KEY environment variable.")

def main():
    st.title("🎬 Movie Recommendation System")

    # Add sidebar for model selection
    st.sidebar.title("Recommendation Settings")
    model_type = st.sidebar.selectbox(
        "Select Recommendation Model",
        ["Content-Based", "Collaborative Filtering", "Hybrid", "Matrix Factorization"]
    )

    # Add explanation of the selected model
    if model_type == "Content-Based":
        st.sidebar.info(
            "Content-Based filtering recommends movies similar to what you've liked before, "
            "based on movie attributes like genre, actors, and directors."
        )
    elif model_type == "Collaborative Filtering":
        st.sidebar.info(
            "Collaborative Filtering recommends movies based on what similar users have liked. "
            "It finds patterns in user behavior to make personalized recommendations."
        )
    elif model_type == "Hybrid":
        st.sidebar.info(
            "Hybrid recommendation combines Content-Based and Collaborative Filtering approaches "
            "to provide more accurate and diverse recommendations."
        )
        # Add sliders for weights
        content_weight = st.sidebar.slider("Content-Based Weight", 0.0, 1.0, 0.5, 0.1)
        collab_weight = 1.0 - content_weight
        st.sidebar.text(f"Collaborative Filtering Weight: {collab_weight:.1f}")
    elif model_type == "Matrix Factorization":
        st.sidebar.info(
            "Matrix Factorization is an advanced technique used by Netflix and YouTube. "
            "It decomposes the user-item interaction matrix to discover latent factors "
            "that explain observed preferences."
        )
        # Add slider for number of factors
        n_factors = st.sidebar.slider("Number of Factors", 10, 100, 50, 5)

    # Get user preferences
    preferences = get_user_preferences()
    if preferences is None:
        st.info("Please select your preferences and click Submit.")
        return

    # Fetch movies
    fetcher = MovieFetcher(API_KEY)
    try:
        movies_df = fetcher.fetch_popular_movies(pages=2)
        if movies_df.empty:
            st.error("No movies found. Please try again later.")
            return
    except Exception as e:
        st.error(f"Error fetching movies: {e}")
        return

    # Ensure required columns exist
    required_columns = {'genres', 'vote_average', 'title'}
    if not required_columns.issubset(movies_df.columns):
        st.error("Unexpected data format from the API.")
        return

    # Generate recommendations based on selected model
    if model_type == "Content-Based":
        model = ContentBasedModel(movies_df)
    elif model_type == "Collaborative Filtering":
        model = CollaborativeFilteringModel(movies_df)
    elif model_type == "Hybrid":
        model = HybridRecommendationModel(
            movies_df, 
            content_weight=content_weight,
            collab_weight=collab_weight
        )
    elif model_type == "Matrix Factorization":
        model = MatrixFactorizationModel(movies_df, n_factors=n_factors)

    recommendations = model.recommend(preferences)

    # Display recommendations
    if recommendations.empty:
        st.warning("No recommendations found based on your preferences.")
    else:
        st.subheader("🎥 Recommended Movies:")
        for _, row in recommendations.iterrows():
            st.markdown(f"**{row['title']}** – ⭐ {row['vote_average']:.1f}")

            # Display genre information if available
            if 'genres' in row and row['genres']:
                genres_str = ", ".join(row['genres'])
                st.text(f"Genres: {genres_str}")

            # Display additional information if available
            if model_type in ["Hybrid", "Matrix Factorization"] and 'genre_match' in row:
                # Check for NaN values before passing to progress bar
                genre_match = row['genre_match']
                if pd.notna(genre_match):  # Only show progress if genre_match is not NaN
                    st.progress(min(genre_match, 3) / 3)  # Normalize to 0-1 range

            st.markdown("---")


if __name__ == '__main__':
    main()
