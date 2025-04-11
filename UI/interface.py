import streamlit as st


def get_user_preferences():
    st.title("🎬 System Rekomendacji Filmów")
    genres = st.multiselect("Wybierz ulubione gatunki:",
                            ["Action", "Comedy", "Drama", "Fantasy", "Horror", "Romance", "Thriller"])
    min_rating = st.slider("Minimalna ocena filmu:", 0.0, 10.0, 7.0)
    release_year = st.slider("Zakres lat wydania:", 1900, 2025, (2000, 2025))

    # Add a submit button
    if st.button("Submit"):
        return {"genres": genres, "min_rating": min_rating, "release_year": release_year}
    return None