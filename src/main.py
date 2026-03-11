from data_prep import load_cv_split, load_item, cleaning_data
from base_recommender import BaseRecommender


if __name__ == "__main__":
    items_df = load_item()
    
    for i in range(1, 6): # 5-fold cross-validation
        train_raw, test_raw = load_cv_split(i)
        
        
        _, train_clean = cleaning_data(items_df, train_raw)
        _, test_clean = cleaning_data(items_df, test_raw)

        # 1. Feed into models
        # 2. Model fitting
        # 3. Evaluation