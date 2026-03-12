from data_prep import load_cv_split, load_item, cleaning_data
from base_recommender import BaseRecommender
from collaborative import CollaborativeRecommender

models = [CollaborativeRecommender("Collaborative", k=3)] # Pass functions for Content and collaborative


if __name__ == "__main__":
    items_df = load_item()
        
    for i in range(1, 6): # 5-fold cross-validation
        train_raw, test_raw = load_cv_split(i)
        
        
        _, train_clean = cleaning_data(items_df, train_raw)
        _, test_clean = cleaning_data(items_df, test_raw)

        for model in models:
            
            # 1. Feed into models
            # Jon: I don't know what we should put here.
            # 2. Model fitting
            # model.fit(train_clean, items_df)
            model.fit(train_clean, items_df)
            # 3. Evaluation
            #model.predict_rating(user_id, item_id)
            model.predict_rating(user_id=196, item_id=302)
            model.recommend(user_id=196, top_n=3) 
            
            pass