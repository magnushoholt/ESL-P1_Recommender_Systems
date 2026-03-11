

from base_recommender import BaseRecommender

# Jeg har lige forsøgt at give en skitse af, hvordan man bruger den abstrakte klasse BaseRecommender, 
# og hvordan man kan implementere en simpel CollaborativeRecommender, der arver fra BaseRecommender.

class CollaborativeRecommender(BaseRecommender):
    def __init__(self, name):
        super().__init__(name)


    def fit(self, train_data, items_df):
        # Implement collaborative filtering logic here
        self.item_df = items_df
        self.train_data = train_data
        pass
    
    def predict_rating(self, user_id, item_id):
        # Implement rating prediction logic here
        pass

    def recommend(self, user_id, top_n=5):
        # Implement recommendation logic here
        pass