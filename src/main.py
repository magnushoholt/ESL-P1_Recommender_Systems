from data_prep import load_cv_split, load_item, cleaning_data
from base_recommender import BaseRecommender
from collaborative import CollaborativeRecommender
from eval import compute_rmse

models = [CollaborativeRecommender("Collaborative", k=3)] # Pass functions for Content and collaborative


if __name__ == "__main__":
    items_df = load_item()
        
    for i in range(1, 6): # 5-fold cross-validation
        print(f"")
        print(f"===============================")
        print(f"--- Fold {i} ---")  # so we can see which fold we are on
        train_raw, test_raw = load_cv_split(i)
        
        
        _, train_clean = cleaning_data(items_df, train_raw)
        _, test_clean = cleaning_data(items_df, test_raw)

        for model in models:
            print(f"Training {model.name}...") # What model we are training
            
            # 1. Feed into models
            # Jon: I don't know what we should put here.
            
            # 2. Model fitting
            model.fit(train_clean, items_df)

            score = model.predict_rating(user_id=196, item_id=302)
            movie_title = items_df.loc[items_df["movie_id"] == 302, "movie_title"].iloc[0]
            print(f"Predicted rating for user 196, film: '{movie_title}' - {score:.0f}")
            
            print(f"")
            n_recs = 3
            print(f"Top {n_recs} recommendations for user 196:")
            recs = model.recommend(user_id=196, top_n=n_recs)
            for rec in recs:
                print(f"  {rec['title']} (predicted: {rec['score']:.0f})")
                for c in rec['contributors']:
                    print(f"    neighbor {c['user_id']} sim={c['similarity']:.2f} rated {c['rating']:.0f}")

            # 3. Evaluate: compute RMSE on the test set and print it
            rmse = compute_rmse(model, test_clean)
            print(f"{model.name} RMSE: {rmse:.4f}")
            
            pass