from data_prep import load_cv_split, load_item, cleaning_data
from base_recommender import BaseRecommender
from collaborative import CollaborativeRecommender
from eval import compute_rmse, compute_single_rmse

# Compare multiple collaborative variants with minimal extra plumbing.
models = [
    CollaborativeRecommender("User-User Cosine", k=3, user_based=True, similarity="cosine"),
    CollaborativeRecommender("Item-Item Cosine", k=3, user_based=False, similarity="cosine"),
    CollaborativeRecommender("User-User Pearson", k=3, user_based=True, similarity="pearson"),
]


def find_true_rating(ratings_df, user_id, item_id):
    """Return the known rating for one (user, item) pair if it exists."""
    match = ratings_df[
        (ratings_df["user_id"] == user_id) &
        (ratings_df["item_id"] == item_id)
    ]
    if match.empty:
        return None
    return float(match.iloc[0]["rating"])


def movie_title_from_id(items_df, item_id):
    """Resolve movie title from item_id, with a safe fallback label."""
    title_match = items_df[items_df["movie_id"] == item_id]["movie_title"]
    if title_match.empty:
        return f"Item {item_id}"
    return title_match.iloc[0]


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
            print(f"")
            print(f"Training {model.name}...")

            # 1. Fit the model to the training data
            model.fit(train_clean, items_df)

            # 2. Predict the rating for a specific (user, item) pair and display it
            item_id = 11
            user_id = 1
            movie_title = movie_title_from_id(items_df, item_id=item_id)
            true_rating = find_true_rating(train_clean, user_id=user_id, item_id=item_id)
            score = model.predict_rating(user_id=user_id, item_id=item_id, display=True)
            if true_rating is None:
                print(f"True rating for user {user_id} on '{movie_title}' in this fold: not available")
            else:
                print(f"True rating for user {user_id} on '{movie_title}' in this fold: {true_rating:.1f}")
                sample_rmse = compute_single_rmse(score, true_rating)
                print(f"Single-sample RMSE: {sample_rmse:.4f}")

            # 3. Generate and display top-N recommendations for the user
            n_recs = 3
            model.recommend(user_id=user_id, top_n=n_recs, display=True)

            # 4. Compute and display the RMSE for the model on the test set
            rmse = compute_rmse(model, test_clean)
            print(f"{model.name} RMSE: {rmse:.4f}")