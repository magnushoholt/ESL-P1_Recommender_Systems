import pandas as pd
from data_prep import load_data, load_item, cleaning_data
from sklearn.metrics.pairwise import cosine_similarity

class Content_recommender_system():
    def __init__(self, item_clean):
        self.items_df = item_clean
        # Unknown is removed since it does not make sense to recommend a movie of unknown genre
        self.items_df = self.items_df.drop(["release_date","video_release_date","IMDb_URL","unknown"], axis=1)
        self.features = self.items_df.iloc[:, 2:]

    
    def recommendation(self, input):
        self.similarity = cosine_similarity(self.features, input).tolist()
        temp = self.items_df.copy()
        temp["Score"] = self.similarity
        self.recom = temp.sort_values(by="Score", ascending=False)

    def get_input_title(self, title):
        idx = self.items_df.index[self.items_df['movie_title'] == title]
        if len(idx) == 0:
            return "Movie not found"
        idx = self.items_df.index.get_loc(idx[0])
        out = self.items_df.iloc[idx:idx+1,2:]
        return out


if __name__ == "__main__":
    data = load_data()
    items = load_item()
    items_clean, _ = cleaning_data(items,data)
    items_rating = Content_recommender_system(items_clean)
    input = items_rating.get_input_title("Toy Story (1995)")
    items_rating.recommendation(input)
    print(items_rating.recom.iloc[0:10,[1,20]])
    