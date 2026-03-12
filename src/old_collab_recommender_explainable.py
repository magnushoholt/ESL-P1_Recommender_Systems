# Collaborative filtering recommender system using user-based approach.

# Surprise is a Python SciKit that comes with various recommender algorithms 
# and similarity metrics to make it easy to build and analyze recommenders.
# pip install scikit-surprise (requires python 3.11 and VS build tools)
# also need a downgraded version of numpuy: pip install numpy==1.26.4 (pain in the ass)

# Simplest to just make a new venv and install:
#python -m pip install --upgrade pip setuptools wheel
#python -m pip install "numpy==1.26.4" "scipy==1.11.4" "cython<3"
#python -m pip install --no-build-isolation --no-cache-dir scikit-surprise

# this one also needs pandas
#python -m pip install pandas


# ---------------- Setup data ----------------

import os
import pandas as pd

# Surprise dataset API and algorithm classes.
from surprise import Dataset
from surprise import KNNWithMeans
from surprise import accuracy
from surprise.model_selection import train_test_split

# Load Surprise's built-in MovieLens 100K dataset as the source data.
# prompt=False avoids interactive terminal prompts if the dataset must be downloaded.
data = Dataset.load_builtin("ml-100k", prompt=False)


def load_movie_titles():
    # Resolve path relative to this file so it works from any working directory.
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "u.item")
    # u.item is pipe-separated; columns 0 and 1 are movie_id and title.
    df = pd.read_csv(data_path, sep="|", header=None, encoding="latin-1", usecols=[0, 1])
    df.columns = ["movie_id", "title"]
    # Surprise stores ids as strings, so we key by string.
    return {str(row.movie_id): row.title for row in df.itertuples()}


# Map item id string -> movie title, used for human-readable output.
movie_titles = load_movie_titles()





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


def explain_prediction(user_id: str, item_id: str, training_set, top_contributors: int = 3):
    """Return the neighbors that contributed most to predicting user_id's rating for item_id.

    Requires algo to already be fitted on training_set.
    Each contributor dict has: user_id, similarity, rating, contribution.
    contribution = similarity * (neighbor_rating - neighbor_mean), same formula as KNNWithMeans.
    """
    try:
        # Translate raw string ids to Surprise's internal integer ids.
        inner_uid = training_set.to_inner_uid(user_id)
        inner_iid = training_set.to_inner_iid(item_id)
    except ValueError:
        # User or item was not in the training set; no explanation possible.
        return []

    # ir[inner_iid] is the list of (inner_uid, rating) for everyone who rated this item.
    raters = training_set.ir[inner_iid]

    candidates = []
    for neighbor_inner_uid, neighbor_rating in raters:
        if neighbor_inner_uid == inner_uid:
            # Skip the target user themselves.
            continue
        # algo.sim is the full n_users x n_users similarity matrix computed during fit.
        sim = algo.sim[inner_uid][neighbor_inner_uid]
        if sim <= 0:
            # Ignore users with zero or negative similarity (no useful signal).
            continue
        candidates.append((neighbor_inner_uid, neighbor_rating, float(sim)))

    # KNNWithMeans selects neighbors by HIGHEST similarity, not by contribution size.
    # We must mirror that same selection here, otherwise our explanation contradicts
    # the actual prediction (it would show different users than the algo used).
    candidates.sort(key=lambda x: x[2], reverse=True)
    # Take the same k neighbors the algo used.
    actual_neighbors = candidates[:algo.k]

    contributors = []
    for neighbor_inner_uid, neighbor_rating, sim in actual_neighbors:
        # algo.means holds each user's average rating, computed during fit.
        neighbor_mean = algo.means[neighbor_inner_uid]
        # Weighted deviation: how much this neighbor pulled the prediction up or down.
        contribution = sim * (neighbor_rating - neighbor_mean)
        contributors.append({
            "user_id": training_set.to_raw_uid(neighbor_inner_uid),
            "similarity": round(sim, 3),
            "rating": neighbor_rating,
            "contribution": round(float(contribution), 3),
        })

    # Show most influential neighbor first.
    contributors.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributors[:top_contributors]





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
    """Predict ratings for all unseen items for a user and return top-N with explanations."""
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

    # Collect predicted scores and explanations for candidate items.
    scored = []
    for item_id in unseen_items:
        # Predict user preference for this unseen item.
        pred = algo.predict(user_id, item_id)
        # Retrieve which neighbors drove this prediction and how much.
        contributors = explain_prediction(user_id, item_id, training_set)
        scored.append({
            "item_id": item_id,
            # Look up human-readable title; fall back to raw id if not found.
            "title": movie_titles.get(item_id, f"Item {item_id}"),
            "score": pred.est,
            "contributors": contributors,
        })

    # Highest predicted ratings first.
    scored.sort(key=lambda x: x["score"], reverse=True)
    # Return only the top N predictions.
    return scored[:top_n]

# ---------------- Present recommendations to the user ----------------


def main():
    # Use IDs that exist in MovieLens 100K (raw ids are strings in Surprise API).
    est = predict_for_user_item("196", "302")
    # Compute basic holdout quality metric.
    rmse = evaluate_with_rmse()
    # Generate top recommendations for the same user, including explanations.
    recs = top_recommendations_for_user("196", top_n=3)

    # Print one direct prediction example.
    print(f"Predicted rating for user 196 on item 302: {est:.3f}")
    # Print RMSE from random train/test split.
    print(f"Holdout RMSE: {rmse:.3f}")

    # Print top-N recommendations with human-readable explanations.
    print("\nTop recommendations for user 196:")
    for rec in recs:
        print(f"\n  '{rec['title']}' (predicted rating: {rec['score']:.2f})")
        if rec["contributors"]:
            print("  Why: Similar users who rated this movie highly:")
            for c in rec["contributors"]:
                direction = "above" if c["contribution"] > 0 else "below"
                print(
                    f"    - User {c['user_id']} "
                    f"(similarity {c['similarity']:.2f}) "
                    f"rated it {c['rating']:.1f} "
                    f"— {direction} their average (contribution: {c['contribution']:+.2f})"
                )
        else:
            print("  Why: No direct neighbor evidence available for this item.")


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

#Top recommendations for user 196:

#  'Conspiracy Theory (1997)' (predicted rating: 5.00)
#  Why: Similar users who rated this movie highly:
#    - User 3 (similarity 1.00) rated it 5.0 — above their average (contribution: +2.20)
#    - User 61 (similarity 1.00) rated it 5.0 — above their average (contribution: +2.05)
#    - User 166 (similarity 1.00) rated it 5.0 — above their average (contribution: +1.45)
#
#  'Close Shave, A (1995)' (predicted rating: 5.00)
#  Why: Similar users who rated this movie highly:
#    - User 865 (similarity 1.00) rated it 5.0 — above their average (contribution: +2.71)
#    - User 822 (similarity 0.99) rated it 5.0 — above their average (contribution: +1.86)
#    - User 359 (similarity 1.00) rated it 5.0 — above their average (contribution: +1.07)
#
#  'Wrong Trousers, The (1993)' (predicted rating: 5.00)
#  Why: Similar users who rated this movie highly:
#    - User 865 (similarity 1.00) rated it 5.0 — above their average (contribution: +2.71)
#    - User 165 (similarity 1.00) rated it 5.0 — above their average (contribution: +1.00)
#    - User 516 (similarity 1.00) rated it 5.0 — above their average (contribution: +0.91)