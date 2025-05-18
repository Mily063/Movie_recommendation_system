import streamlit as st
import random


def get_user_preferences():
    basic_tab, advanced_tab = st.tabs(["Basic Preferences", "Advanced Preferences"])

    with basic_tab:
        genres = st.multiselect(
            "Select your favorite genres:",
            [
                "Action",
                "Adventure",
                "Animation",
                "Comedy",
                "Crime",
                "Documentary",
                "Drama",
                "Family",
                "Fantasy",
                "History",
                "Horror",
                "Music",
                "Mystery",
                "Romance",
                "Science Fiction",
                "TV Movie",
                "Thriller",
                "War",
                "Western",
            ],
        )

        min_rating = st.slider("Minimum movie rating:", 0.0, 10.0, 7.0)
        release_year = st.slider("Release year range:", 1900, 2025, (2000, 2025))

    with advanced_tab:
        st.subheader("Personalization Settings")

        user_id = st.number_input(
            "User ID (for personalization):",
            min_value=1,
            max_value=100,
            value=random.randint(1, 100),
            help="In a real system, this would be your account ID. For demo purposes, a random ID is generated.",
        )

        st.write("Movie Mood Preferences:")
        mood_cols = st.columns(3)
        with mood_cols[0]:
            want_popular = st.checkbox("Popular movies", value=True)
        with mood_cols[1]:
            want_recent = st.checkbox("Recent releases", value=True)
        with mood_cols[2]:
            want_diverse = st.checkbox("Diverse recommendations", value=True)

        duration_preference = st.select_slider(
            "Preferred movie length:",
            options=[
                "Short (<90 min)",
                "Medium (90-120 min)",
                "Long (>120 min)",
                "No preference",
            ],
            value="No preference",
        )

        language_preference = st.multiselect(
            "Preferred languages:",
            [
                "English",
                "Spanish",
                "French",
                "German",
                "Japanese",
                "Korean",
                "Chinese",
                "Other",
            ],
            default=["English"],
        )

    if st.button("Get Recommendations"):
        preferences = {
            "genres": genres,
            "min_rating": min_rating,
            "release_year": release_year,
            "user_id": user_id,
        }

        if "want_popular" in locals():
            preferences.update(
                {
                    "want_popular": want_popular,
                    "want_recent": want_recent,
                    "want_diverse": want_diverse,
                    "duration_preference": duration_preference,
                    "language_preference": language_preference,
                }
            )

        return preferences

    return None
