import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from data_prep import load_data, load_item, cleaning_data, matrix_data
from base_recommender import BaseRecommender

class Content_recommender_system(BaseRecommender):
    def __init__(self, name):
        super().__init__(name)
    
    def _user_genre(self, features, train_data):

        user_item_matrix = matrix_data(train_data)
        
        user_item_matrix = user_item_matrix.reindex(columns=features.index)

        user_item_matrix = user_item_matrix.fillna(0)
 
        user_item_matrix_np = user_item_matrix.to_numpy()
        features_np = features.to_numpy()

        user_genre = (user_item_matrix_np[:, :, None] * features_np[None, :, :])
        user_genre = user_genre.sum(axis=1) # Sums up all movies

        row_max = user_genre.max(axis=1, keepdims=True)
        user_genre = np.divide(user_genre, row_max, out=np.zeros_like(user_genre), where=row_max!=0)

        user_genre = pd.DataFrame(
            user_genre,
            index=user_item_matrix.index,   # user_id
            columns=features.columns        # genre names
        )

        return user_genre
    
    def _average_rating(self, train_data, bias):

        user_item_matrix = matrix_data(train_data)
        number_of_ratings = user_item_matrix.count()
        total_rating = user_item_matrix.sum()
        mean_rating = total_rating/(number_of_ratings+bias)

        mean_rating = mean_rating.reindex(index=self.features.index).fillna(0)

        mean_rating = mean_rating/mean_rating.max()

        self.features["Rating"] = mean_rating

        self.user_max = mean_rating.max()

        self.user_genre["Rating"] = self.user_max



    def fit(self, train_data, item_df, include_rating = False, rating_bias = 0):
        self.items_df = item_df.sort_values(by="movie_id").set_index("movie_id")
        self.features = self.items_df.iloc[:, 5:]
        self.include_rating = include_rating

        self.user_genre = self._user_genre(self.features, train_data)
        
        if self.include_rating:
            self._average_rating(train_data, rating_bias)

    def _compute_similarity(self,X,Y):
        similarity = cosine_similarity(X, Y).tolist()
        final_score = self.items_df.copy()
        final_score["Score"] = similarity
        return final_score

    def predict_rating(self, user_id, item_id):
        user_chosen = self.user_genre.loc[[user_id]]
        item_chosen = self.features.loc[[item_id]]
        #print("User")
        #print(user_chosen.head())
        #print("Item")
        #print(item_chosen.head())
        out = cosine_similarity(user_chosen,item_chosen)[0][0]
        return out*4+1
    
    def recommend(self, user_id, top_n=5):
        user_chosen = self.user_genre.loc[[user_id]]
        similarity = self._compute_similarity(self.features, user_chosen)
        sorted = similarity.sort_values(by="Score", ascending=False)
        out = sorted.iloc[0:top_n,[0,-1]]
        return out


if __name__ == "__main__":
    data = load_data()
    items = load_item()
    items_clean, data = cleaning_data(items,data)
    items_rating = Content_recommender_system("Content")
    items_rating.fit(data,items_clean, False)

    target_user = 1
    target_item = 1
    print(items_rating.items_df.head())
    out = items_rating.recommend(target_user)
    print(out)
    