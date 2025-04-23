# System Rekomendacji Filmów
## Prezentacja projektu

---

## Agenda
1. Wprowadzenie do projektu
2. Zaawansowane metody dziedziczenia
3. Elementy uczenia maszynowego
4. Techniki rekomendacji stosowane przez Netflix i YouTube
5. Struktura projektu
6. Demonstracja systemu
7. Plany rozwoju

---

## Wprowadzenie do projektu

### Cel projektu
- Stworzenie zaawansowanego systemu rekomendacji filmów
- Implementacja różnych algorytmów rekomendacji
- Zastosowanie technik używanych przez platformy streamingowe

### Główne funkcjonalności
- Rekomendacje oparte na zawartości
- Rekomendacje oparte na współpracy
- Rekomendacje hybrydowe
- Personalizacja preferencji użytkownika

---

## Zaawansowane metody dziedziczenia

### Klasy abstrakcyjne (Abstract Base Classes)
- Wykorzystanie modułu ABC w Pythonie
- Definiowanie abstrakcyjnych interfejsów dla modeli rekomendacji
- Zapewnienie spójności implementacji różnych modeli

```python
from abc import ABC, abstractmethod

class RecommendationModel(ABC):
    @abstractmethod
    def recommend(self, preferences: dict, n: int = 5):
        pass
```

---

## Zaawansowane metody dziedziczenia (cd.)

### Kompozycja zamiast dziedziczenia
- Model hybrydowy wykorzystuje kompozycję do łączenia różnych strategii rekomendacji
- Elastyczność w konfiguracji i rozszerzaniu systemu
- Unikanie problemów związanych z wielokrotnym dziedziczeniem

```python
class HybridRecommendationModel(RecommendationModel):
    def __init__(self, movies_df, ratings_df=None, content_weight=0.5, collab_weight=0.5):
        self.content_model = ContentBasedModel(movies_df)
        self.collaborative_model = CollaborativeFilteringModel(movies_df, ratings_df)
```

---

## Zaawansowane metody dziedziczenia (cd.)

### Nadpisywanie metod (Method Overriding)
- Każdy model rekomendacji implementuje własną wersję metody `recommend`
- Dostosowanie zachowania do specyfiki danego algorytmu
- Zachowanie spójnego interfejsu dla wszystkich modeli

```python
def recommend(self, preferences: dict, n: int = 5):
    # Implementacja specyficzna dla danego modelu
    # ...
    return recommendations
```

---

## Elementy uczenia maszynowego

### Filtrowanie oparte na zawartości (Content-Based Filtering)
- Rekomenduje filmy podobne do tych, które użytkownik lubił wcześniej
- Bazuje na atrybutach filmów (gatunki, aktorzy, reżyserzy)
- Analizuje właściwości filmów i rekomenduje pozycje o podobnych cechach

```python
filtered = self.movies_df[
    (self.movies_df['genres'].apply(lambda g: any(genre in g for genre in preferred_genres))) &
    (self.movies_df['vote_average'] >= min_rating)
]
```

---

## Elementy uczenia maszynowego (cd.)

### Filtrowanie kolaboratywne (Collaborative Filtering)
- Rekomenduje filmy na podstawie preferencji podobnych użytkowników
- Znajduje wzorce w zachowaniach użytkowników
- Nie wymaga zrozumienia zawartości filmów

```python
# Znajdź podobnych użytkowników
similar_users = self._get_similar_users(user_id)
        
# Pobierz filmy wysoko ocenione przez podobnych użytkowników
similar_users_ratings = self.user_item_matrix.loc[similar_users]
mean_ratings = similar_users_ratings.mean()
```

---

## Elementy uczenia maszynowego (cd.)

### Faktoryzacja macierzy (Matrix Factorization)
- Rozkłada macierz interakcji użytkownik-film na czynniki ukryte
- Odkrywa ukryte wzorce w preferencjach użytkowników
- Technika spopularyzowana przez Netflix

```python
def _perform_svd(self):
    # Wykonaj SVD (Singular Value Decomposition)
    matrix = self.user_item_matrix.fillna(0).values
    U, sigma, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    # Ogranicz do n_factors
    U = U[:, :self.n_factors]
    sigma = sigma[:self.n_factors]
    Vt = Vt[:self.n_factors, :]
    
    return U, sigma, Vt
```

---

## Elementy uczenia maszynowego (cd.)

### Rekomendacje hybrydowe (Hybrid Recommendation)
- Łączy filtrowanie oparte na zawartości i filtrowanie kolaboratywne
- Wykorzystuje mocne strony obu metod
- Zapewnia bardziej dokładne i zróżnicowane rekomendacje

```python
def recommend(self, preferences: dict, n: int = 5):
    # Pobierz rekomendacje z każdego modelu
    content_recs = self.content_model.recommend(preferences, n=n*2)
    collab_recs = self.collaborative_model.recommend(preferences, n=n*2)
    
    # Połącz rekomendacje z wagami
    combined_recs = self._combine_recommendations(content_recs, collab_recs)
    
    # Zastosuj zwiększenie różnorodności
    diverse_recs = self._enhance_diversity(combined_recs, n)
    
    return diverse_recs.head(n)
```

---

## Techniki rekomendacji stosowane przez Netflix i YouTube

### Personalizacja
- Dostosowywanie rekomendacji do indywidualnych preferencji użytkownika
- Uwzględnianie gatunków, ocen, lat wydania
- Możliwość określenia szczegółowych preferencji

```python
# Interfejs użytkownika do zbierania preferencji
genres = st.multiselect(
    "Wybierz ulubione gatunki:",
    ["Akcja", "Przygodowy", "Animacja", "Komedia", "Kryminał", ...]
)

min_rating = st.slider("Minimalna ocena filmu:", 0.0, 10.0, 7.0)
release_year = st.slider("Zakres lat wydania:", 1900, 2025, (2000, 2025))
```

---

## Techniki rekomendacji stosowane przez Netflix i YouTube (cd.)

### Zwiększanie różnorodności (Diversity Enhancement)
- Zapobieganie efektowi "bańki filtrującej"
- Algorytm zachłanny do zapewnienia różnorodności rekomendacji
- Użytkownicy otrzymują zróżnicowane propozycje

```python
def _enhance_diversity(self, recommendations, n):
    # Zacznij od najwyżej ocenianej pozycji
    diverse_indices = [0]
    candidate_indices = list(range(1, len(recommendations)))
    
    # Algorytm zachłanny do maksymalizacji różnorodności
    while len(diverse_indices) < n and candidate_indices:
        max_diversity_idx = None
        max_diversity = -1
        
        for idx in candidate_indices:
            # Oblicz różnorodność jako sumę różnic gatunków
            diversity = self._calculate_diversity(recommendations.iloc[idx], 
                                                recommendations.iloc[diverse_indices])
            
            if diversity > max_diversity:
                max_diversity = diversity
                max_diversity_idx = idx
        
        if max_diversity_idx is not None:
            diverse_indices.append(max_diversity_idx)
            candidate_indices.remove(max_diversity_idx)
        else:
            break
    
    return recommendations.iloc[diverse_indices]
```

---

## Techniki rekomendacji stosowane przez Netflix i YouTube (cd.)

### Podejście hybrydowe z wagami (Weighted Hybrid Approach)
- Możliwość dostosowania wpływu różnych strategii rekomendacji
- Użytkownicy mogą regulować wagi dla filtrowania opartego na zawartości i kolaboratywnego
- Dostrajanie strategii rekomendacji

```python
# Dodaj suwaki dla wag
content_weight = st.sidebar.slider("Waga filtrowania opartego na zawartości", 0.0, 1.0, 0.5, 0.1)
collab_weight = 1.0 - content_weight
st.sidebar.text(f"Waga filtrowania kolaboratywnego: {collab_weight:.1f}")
```

---

## Techniki rekomendacji stosowane przez Netflix i YouTube (cd.)

### Faktoryzacja macierzy (Matrix Factorization)
- Podstawowa technika używana w systemie rekomendacji Netflix
- Odkrywanie ukrytych wzorców w preferencjach użytkowników
- Przewidywanie ocen na podstawie rozkładu macierzy

```python
def _predict_ratings(self, user_idx):
    """Przewiduj oceny dla użytkownika na podstawie sfaktoryzowanych macierzy."""
    user_factors = self.U[user_idx, :]
    predicted_ratings = np.dot(np.dot(user_factors, np.diag(self.sigma)), self.Vt)
    return predicted_ratings
```

---

## Struktura projektu

### Główne komponenty
- **Recommender/**: Zawiera implementacje modeli rekomendacji
  - **base.py**: Definiuje abstrakcyjną klasę bazową dla wszystkich modeli
  - **content_based.py**: Implementuje filtrowanie oparte na zawartości
  - **collaborative_filtering.py**: Implementuje filtrowanie kolaboratywne
  - **hybrid.py**: Implementuje rekomendacje hybrydowe i modele faktoryzacji macierzy
  - **fetcher.py**: Obsługuje pobieranie danych o filmach z API
- **UI/**: Zawiera komponenty interfejsu użytkownika
  - **interface.py**: Obsługuje zbieranie preferencji użytkownika
- **main.py**: Główny punkt wejścia aplikacji

---

## Struktura projektu (cd.)

### Przepływ danych
1. Użytkownik określa swoje preferencje
2. System pobiera popularne filmy z API
3. Wybrany model rekomendacji przetwarza dane
4. System prezentuje rekomendowane filmy

```python
def main():
    # Pobierz preferencje użytkownika
    preferences = get_user_preferences()
    
    # Pobierz filmy
    fetcher = MovieFetcher(API_KEY)
    movies_df = fetcher.fetch_popular_movies(pages=2)
    
    # Wygeneruj rekomendacje na podstawie wybranego modelu
    model = HybridRecommendationModel(movies_df, content_weight=0.5, collab_weight=0.5)
    recommendations = model.recommend(preferences)
    
    # Wyświetl rekomendacje
    # ...
```

---

## Demonstracja systemu

### Interfejs użytkownika
- Oparty na bibliotece Streamlit
- Intuicyjny i responsywny
- Możliwość wyboru modelu rekomendacji
- Szczegółowe ustawienia preferencji

### Wybór modelu rekomendacji
- Filtrowanie oparte na zawartości
- Filtrowanie kolaboratywne
- Model hybrydowy
- Faktoryzacja macierzy

---

## Demonstracja systemu (cd.)

### Podstawowe preferencje
- Wybór ulubionych gatunków
- Minimalna ocena filmu
- Zakres lat wydania

### Zaawansowane preferencje
- ID użytkownika (do personalizacji)
- Preferencje nastroju filmu
- Preferowana długość filmu
- Preferowane języki

---

## Demonstracja systemu (cd.)

### Wyświetlanie rekomendacji
- Tytuł filmu
- Ocena
- Gatunki
- Dodatkowe informacje (np. dopasowanie gatunku)

```python
st.subheader("🎥 Rekomendowane filmy:")
for _, row in recommendations.iterrows():
    st.markdown(f"**{row['title']}** – ⭐ {row['vote_average']:.1f}")
    
    # Wyświetl informacje o gatunku, jeśli dostępne
    if 'genres' in row and row['genres']:
        genres_str = ", ".join(row['genres'])
        st.text(f"Gatunki: {genres_str}")
    
    # Wyświetl dodatkowe informacje, jeśli dostępne
    if model_type in ["Hybrid", "Matrix Factorization"] and 'genre_match' in row:
        genre_match = row['genre_match']
        if pd.notna(genre_match):  # Tylko pokaż pasek postępu, jeśli genre_match nie jest NaN
            st.progress(min(genre_match, 3) / 3)  # Normalizuj do zakresu 0-1
```

---

## Plany rozwoju

### Przyszłe ulepszenia
- Implementacja modeli rekomendacji opartych na głębokim uczeniu
- Dodanie wsparcia dla ocen i opinii użytkowników
- Uwzględnienie większej ilości metadanych filmów (aktorzy, reżyserzy itp.)
- Implementacja aktualizacji rekomendacji w czasie rzeczywistym na podstawie interakcji użytkownika

---

## Podsumowanie

### Kluczowe osiągnięcia projektu
- Implementacja zaawansowanych metod dziedziczenia
- Wykorzystanie technik uczenia maszynowego
- Zastosowanie metod rekomendacji używanych przez Netflix i YouTube
- Stworzenie elastycznego i rozszerzalnego systemu

### Zdobyta wiedza
- Projektowanie obiektowe w Pythonie
- Algorytmy rekomendacji
- Techniki uczenia maszynowego
- Tworzenie interaktywnych aplikacji webowych

---

## Pytania?

Dziękuję za uwagę!