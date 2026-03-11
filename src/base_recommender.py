from abc import ABC, abstractmethod


class BaseRecommender:
    @abstractmethod
    def fit(self, train_data):
        pass

    @abstractmethod
    def recommend(self, user_id, top_n=5):
        """
        Should return a list of (item_id, score, explanation).
        This is millers way of adressing "cause of a decision"
        """
        pass
    