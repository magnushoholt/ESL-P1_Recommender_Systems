from data_prep import load_cv_split, load_item, cleaning_data
from base_recommender import BaseRecommender
from collaborative import CollaborativeRecommender
from content import Content_recommender_system
from eval import compute_rmse, compute_single_rmse

# Compare multiple recommender variants:
# Collaborative: user-user and item-item with different k and similarity metrics.
# Content-based: genre-based with optional rating weighting.
models = [
    CollaborativeRecommender("User-User Cosine k=3",  k=3,  user_based=True,  similarity="cosine"),
    CollaborativeRecommender("User-User Cosine k=10", k=10, user_based=True,  similarity="cosine"),
    CollaborativeRecommender("User-User Cosine k=25", k=25, user_based=True,  similarity="cosine"),
    CollaborativeRecommender("Item-Item Cosine k=3",  k=3,  user_based=False, similarity="cosine"),
    CollaborativeRecommender("User-User Euclidean k=3", k=3,  user_based=True,  similarity="euclidean"),
    Content_recommender_system("Content (Genres Only)"),
    Content_recommender_system("Content (Genres + Rating bias=0)"),
    Content_recommender_system("Content (Genres + Rating bias=1)"),
    Content_recommender_system("Content (Genres + Rating bias=5)"),
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


def display_prediction_result(user_id, item_id, movie_title, predicted_score, true_rating):
    """Display prediction result with true rating comparison and RMSE if available."""
    if predicted_score is None:
        print(f"Predicted rating for user {user_id}, film: '{movie_title}' - unavailable")
    else:
        print(f"Predicted rating for user {user_id}, film: '{movie_title}' - {predicted_score:.1f}")

    if true_rating is None:
        print(f"True rating for user {user_id} on '{movie_title}' in this fold: not available")
    else:
        print(f"True rating for user {user_id} on '{movie_title}' in this fold: {true_rating:.1f}")
        sample_rmse = compute_single_rmse(predicted_score, true_rating)
        print(f"Single-sample RMSE: {sample_rmse:.4f}")


def _to_scalar_score(value):
    """Convert scalar-or-list score values into a printable float."""
    # Content recommender stores cosine output as list-like values, e.g. [0.73].
    while isinstance(value, (list, tuple)) and len(value) > 0:
        value = value[0]
    return float(value)


def display_recommendations(model, user_id, top_n, items_df):
    """Display top-N recommendations, handling both collaborative and content models."""
    if isinstance(model, CollaborativeRecommender):
        # Ask collaborative model for structured data and display from main.
        recs = model.recommend(user_id=user_id, top_n=top_n, display=False)
        print(f"Top {top_n} recommendations for user {user_id}:")
        if not recs:
            print("  No recommendations available.")
            return

        for rec in recs:
            print(f"  {rec['title']} (predicted: {rec['score']:.1f})")
            source_label = "user" if model.user_based else "item"
            for c in rec["contributors"]:
                print(
                    f"    neighbor {source_label} {c['source_id']} "
                    f"sim={c['similarity']:.5f} "
                    f"rated {c['rating']:.1f} "
                    f"contrib={c['contribution']:+.3f}"
                )
    else:
        # Content model returns dataframe; format it ourselves
        recs_df = model.recommend(user_id=user_id, top_n=top_n)
        print(f"Top {top_n} recommendations for user {user_id}:")
        for idx, (movie_id, row) in enumerate(recs_df.iterrows(), 1):
            score = _to_scalar_score(row["Score"])
            title = movie_title_from_id(items_df, movie_id)
            print(f"  {idx}. {title} (score: {score:.4f})")

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
            if isinstance(model, Content_recommender_system):
                # Content models use include_rating and rating_bias parameters
                include_rating = "Rating" in model.name
                rating_bias = 5 if "bias=5" in model.name else (1 if "bias=1" in model.name else 0)
                model.fit(train_clean, items_df, include_rating=include_rating, rating_bias=rating_bias)
            else:
                # Collaborative models
                model.fit(train_clean, items_df)

            # 2. Predict the rating for a specific (user, item) pair and display it
            item_id = 11
            user_id = 1
            movie_title = movie_title_from_id(items_df, item_id=item_id)
            true_rating = find_true_rating(train_clean, user_id=user_id, item_id=item_id)
            score = model.predict_rating(user_id=user_id, item_id=item_id)
            score = None if score is None else _to_scalar_score(score)
            display_prediction_result(user_id, item_id, movie_title, score, true_rating)

            # 3. Generate and display top-N recommendations for the user
            n_recs = 3
            display_recommendations(model, user_id, n_recs, items_df)

            # 4. Compute and display the RMSE for the model on the test set
            rmse = compute_rmse(model, test_clean)
            print(f"{model.name} RMSE: {rmse:.4f}")