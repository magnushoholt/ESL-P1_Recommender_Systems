from abc import ABC, abstractmethod


class BaseRecommender:
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def fit(self, train_data, item_df):
        """Fit the model to the training data"""
        pass

    def predict_rating(self, user_id, item_id):
        """Returns a predicted rating (1-5) for accuracy evaluation"""
        pass

    @abstractmethod
    def recommend(self, user_id, top_n=5):
        """
        Should return a list of (item_id, score, explanation).
        This is millers way of adressing "cause of a decision"
        Example:
        - Returns a list of dictionaries:
        - [{'item_id': 101, 'score': 4.5, 'explanation': 'Because you liked Byggemand Bob'}]
        """
        pass
