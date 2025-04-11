import os

import streamlit as st
from Recommender.content_based import ContentBasedModel
from Recommender.fetcher import MovieFetcher
from UI.interface import get_user_preferences

API_KEY = os.getenv("TMDB_API_KEY")

if not API_KEY:
    raise ValueError("API key not found. Please set the TMDB_API_KEY environment variable.")

def main():
    st.title("🎬 Movie Recommendation System")

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

    # Generate recommendations
    model = ContentBasedModel(movies_df)
    recommendations = model.recommend(preferences)

    # Display recommendations
    if recommendations.empty:
        st.warning("No recommendations found based on your preferences.")
    else:
        st.subheader("🎥 Recommended Movies:")
        for _, row in recommendations.iterrows():
            st.markdown(f"**{row['title']}** – ⭐ {row['vote_average']:.1f}")


if __name__ == '__main__':
    main()