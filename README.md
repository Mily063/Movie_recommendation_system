# Movie Recommendation System

A sophisticated movie recommendation system that demonstrates advanced inheritance methods and incorporates machine learning techniques used by platforms like YouTube and Netflix.

## Features

### Advanced Inheritance Methods
- **Abstract Base Classes**: Uses Python's ABC module to define abstract interfaces for recommendation models
- **Composition over Inheritance**: The hybrid recommendation model uses composition to combine different recommendation strategies
- **Method Overriding**: Each recommendation model implements its own version of the `recommend` method

### Machine Learning Elements
- **Content-Based Filtering**: Recommends movies based on movie attributes and user preferences
- **Collaborative Filtering**: Recommends movies based on similar users' preferences
- **Matrix Factorization**: Uses Singular Value Decomposition (SVD) to discover latent factors in user-movie interactions
- **Hybrid Recommendation**: Combines multiple recommendation approaches for better results

### Netflix/YouTube-style Recommendation Techniques
- **Personalization**: Tailors recommendations to individual user preferences
- **Diversity Enhancement**: Ensures recommendations aren't too similar to each other
- **Weighted Hybrid Approach**: Allows adjusting the influence of different recommendation strategies
- **Matrix Factorization**: Implements the core technique behind Netflix's recommendation system

## Project Structure

- **Recommender/**: Contains recommendation model implementations
  - **base.py**: Defines the abstract base class for all recommendation models
  - **content_based.py**: Implements content-based filtering
  - **collaborative_filtering.py**: Implements collaborative filtering
  - **hybrid.py**: Implements hybrid recommendation and matrix factorization models
  - **fetcher.py**: Handles fetching movie data from the API
- **UI/**: Contains user interface components
  - **interface.py**: Handles user preference collection
- **main.py**: Main application entry point

## How It Works

### Content-Based Filtering
Content-based filtering recommends movies similar to what you've liked before, based on movie attributes like genre, actors, and directors. It analyzes the attributes of movies and recommends items with similar properties.

### Collaborative Filtering
Collaborative filtering recommends movies based on what similar users have liked. It finds patterns in user behavior to make personalized recommendations without needing to understand the content of the items.

### Matrix Factorization
Matrix factorization decomposes the user-item interaction matrix to discover latent factors that explain observed preferences. This technique, popularized by Netflix, can uncover hidden patterns in user preferences.

### Hybrid Recommendation
The hybrid approach combines content-based and collaborative filtering to leverage the strengths of both methods. It can provide more accurate and diverse recommendations by considering both item attributes and user behavior patterns.

## Advanced Features

### Diversity Enhancement
The system uses a greedy algorithm to ensure diversity in recommendations, preventing the "filter bubble" effect where users only see similar content.

### Personalization
Users can specify detailed preferences including genres, rating thresholds, release years, and even mood preferences to get highly personalized recommendations.

### Weighted Recommendations
In the hybrid model, users can adjust the weights given to content-based and collaborative filtering approaches, allowing for fine-tuning of the recommendation strategy.

## Getting Started

1. Set the TMDB API key as an environment variable:
   ```
   export TMDB_API_KEY="your_api_key_here"
   ```

2. Install the required dependencies:
   ```
   pip install streamlit pandas numpy scikit-learn requests
   ```

3. Run the application:
   ```
   streamlit run main.py
   ```

4. Select your preferences and recommendation model in the UI to get personalized movie recommendations.

## Future Enhancements

- Implement deep learning-based recommendation models
- Add support for user ratings and feedback
- Incorporate more movie metadata (actors, directors, etc.)
- Implement real-time recommendation updates based on user interactions