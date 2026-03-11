# Author: Jontan Blajet
# Collaborative filtering recommender system using user-based approach.

# Surprise is a Python SciKit that comes with various recommender algorithms 
# and similarity metrics to make it easy to build and analyze recommenders.
# pip install scikit-surprise (requires python 3.11 and VS build tools)
# also need a downgraded version of numpuy: pip install numpy==1.26.4 (pain in the ass)

# Simplest to just make a new venv and install:
#python -m pip install --upgrade pip setuptools wheel
#python -m pip install "numpy==1.26.4" "scipy==1.11.4" "cython<3"
#python -m pip install --no-build-isolation --no-cache-dir scikit-surprise

# ---------------- Setup data ----------------

# Surprise dataset API and algorithm classes.
from surprise import Dataset
from surprise import KNNWithMeans
from surprise import accuracy
from surprise.model_selection import train_test_split

# Load Surprise's built-in MovieLens 100K dataset as the source data.
# prompt=False avoids interactive terminal prompts if the dataset must be downloaded.
data = Dataset.load_builtin("ml-100k", prompt=False)





# ---------------- Find similar users ----------------

## Focus only on users who have interacted with the same items
## (ignore other info about users or items, like age, genre, etc.)

## Measure similarity with euclidean or cosine distance
## spatial.distance.euclidean(a, b)
## spatial.distance.cosine(a, b)

## For each user, find the top N most similar users (neighbors)

## Remove bias by normalizing ratings (subtract user mean from each rating)

# For user-based collaborative filtering with cosine similarity.
sim_options = {
    "name": "cosine",  # Similarity metric between users.
    "user_based": True,  # True means user-user similarity (not item-item).
}

# k controls how many nearest neighbors are used for predictions.
algo = KNNWithMeans(k=3, sim_options=sim_options)







# ---------------- Predict ratings for items the user hasn't interacted with ----------------

## Predict rating R for item I by taking average of top N similar neighbors' ratings for item I
## Use weighted ratings, where weights are the similarity scores between the user and the neighbors
## Divide by sum of weights to normalize the prediction


def predict_for_user_item(user_id: str, item_id: int) -> float:
    """Fit on all available data and predict one user-item rating."""
    # Convert the whole dataset into Surprise's internal trainset object.
    training_set = data.build_full_trainset()
    # Learn user means and user-user similarity statistics.
    algo.fit(training_set)
    # Predict rating for one (user, item) pair.
    prediction = algo.predict(user_id, item_id)
    # Return only the estimated rating value.
    return prediction.est


# ---------------- Evaluate the model using RMSE ----------------


def evaluate_with_rmse() -> float:
    """Split data, fit model, and return RMSE on the holdout test set."""
    # Create a random train/test split from the MovieLens dataset.
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
    # Fit on training set only.
    algo.fit(trainset)
    # Predict ratings for all user-item pairs in the test set.
    predictions = algo.test(testset)
    # Compute Root Mean Squared Error (lower is better).
    return accuracy.rmse(predictions, verbose=False)

# ---------------- Filter top ratings ----------------


def top_recommendations_for_user(user_id: str, top_n: int = 3):
    """Predict ratings for all unseen items for a user and return top-N."""
    # Build full trainset so we can inspect seen/unseen items for the target user.
    training_set = data.build_full_trainset()
    # Fit model on all observed ratings.
    algo.fit(training_set)

    # Surprise uses "raw" ids (original ids) and "inner" ids (internal integer ids).
    # Here we collect all item raw ids from the training set.
    all_items = [training_set.to_raw_iid(inner_iid) for inner_iid in training_set.all_items()]

    # If user is unknown to the trainset, we cannot compute seen items reliably.
    if user_id not in training_set._raw2inner_id_users:
        return []

    # Convert user raw id to internal id.
    inner_uid = training_set.to_inner_uid(user_id)
    # Build set of item raw ids the user has already rated.
    seen_items = {
        training_set.to_raw_iid(inner_iid)
        for (inner_iid, _) in training_set.ur[inner_uid]
    }
    # Candidates are items the user has not rated yet.
    unseen_items = [item for item in all_items if item not in seen_items]

    # Collect predicted scores for candidate items.
    scored = []
    for item_id in unseen_items:
        # Predict user preference for this unseen item.
        pred = algo.predict(user_id, item_id)
        # Save tuple of (item, predicted_score).
        scored.append((item_id, pred.est))

    # Highest predicted ratings first.
    scored.sort(key=lambda x: x[1], reverse=True)
    # Return only the top N predictions.
    return scored[:top_n]

# ---------------- Present recommendations to the user ----------------


def main():
    # Use IDs that exist in MovieLens 100K (raw ids are strings in Surprise API).
    est = predict_for_user_item("196", "302")
    # Compute basic holdout quality metric.
    rmse = evaluate_with_rmse()
    # Generate top recommendations for the same user.
    recs = top_recommendations_for_user("196", top_n=3)

    # Print one direct prediction example.
    print(f"Predicted rating for user 196 on item 302: {est:.3f}")
    # Print RMSE from random train/test split.
    print(f"Holdout RMSE: {rmse:.3f}")
    # Print top-N recommendation list.
    print("Top recommendations for user 196 (item_id, predicted_rating):")
    for item_id, score in recs:
        # Print each recommendation row.
        print(f"  item {item_id}: {score:.3f}")


if __name__ == "__main__":
    main()



# ---------------- Notes ----------------

# - Collaborative filtering can be user-based or item-based. User-based finds similar users, while item-based finds similar items.

# Matrix factorization can improve results, but reduce explainability.
# by reducing to latent factor, it makes the recommendations less interpretable.
# so we don't do that.



# Output from running this file:
#Predicted rating for user 196 on item 302: 3.289
#Holdout RMSE: 1.082
#Top recommendations for user 196 (item_id, predicted_rating):
#  item 328: 5.000
#  item 408: 5.000
#  item 169: 5.000