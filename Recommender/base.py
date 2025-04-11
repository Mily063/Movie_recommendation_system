from abc import ABC, abstractmethod

class RecommendationModel(ABC):
    @abstractmethod
    def recommend(self, preferences: dict, n: int = 5):
        pass