import numpy as np
import pandas as pd
from base_recommender import BaseRecommender


class CollaborativeRecommender(BaseRecommender):
    """User-based collaborative filtering recommender with explainable predictions.

    How it works:
      1. fit()           — build a user x item rating matrix and precompute user similarities.
      2. predict_rating()— estimate what rating a user would give an unseen movie.
      3. recommend()     — return the top-N unseen movies with an explanation for each.

    The prediction formula (KNNWithMeans) is:
        predicted = user_mean + Σ(sim * (neighbor_rating - neighbor_mean)) / Σ(sim)

    Mean-centering removes the bias of users who habitually rate high or low.
    """

    def __init__(self, name, k=3):
        """Store the model name and the number of nearest neighbors k.

        Parameters
        ----------
        name : str
            A label for this model instance, e.g. "Collaborative".
            Useful when comparing multiple models side by side.
        k : int, optional (default=3)
            How many similar users to consult when predicting a rating.
            A higher k uses more neighbors (smoother but less personal).
            A lower k relies on fewer, very similar users (more personal but noisier).

            You can set this from main.py when creating the model:
                CollaborativeRecommender("Collaborative")       → k=3 (default)
                CollaborativeRecommender("Collaborative", 5)    → k=5
        """
        super().__init__(name)
        self.k = k                     # how many similar users to consult per prediction
        self.user_item_matrix = None   # DataFrame: rows=users, cols=items, values=ratings (NaN if unseen)
        self.user_means = None         # Series: average rating per user
        self.similarity_matrix = None  # DataFrame: cosine similarity between every pair of users
        self.items_df = None           # DataFrame with movie_id and movie_title columns

    # ------------------------------------------------------------------ #
    #  Public interface (called from main.py / evaluation code)           #
    # ------------------------------------------------------------------ #

    def fit(self, train_data, items_df):
        """Train the model on a ratings DataFrame.

        Parameters
        ----------
        train_data : DataFrame with columns [user_id, item_id, rating, ...]
        items_df   : DataFrame with columns [movie_id, movie_title, ...] for title lookup
        """
        self.items_df = items_df

        # Build the user×item matrix (one row per user, one column per movie)
        self.user_item_matrix = self._build_user_item_matrix(train_data)

        # Compute each user's average rating so we can mean-center later
        self.user_means = self._compute_user_means(self.user_item_matrix)

        # Precompute cosine similarity between every pair of users
        self.similarity_matrix = self._compute_similarity_matrix(
            self.user_item_matrix, self.user_means
        )

    def predict_rating(self, user_id, item_id):
        """Return the predicted rating (1-5) that user_id would give item_id.

        Returns None if the user or item is not in the training data.
        """
        return self._predict_single(user_id, item_id)

    def recommend(self, user_id, top_n=5):
        """Return the top_n unseen movies for user_id, each with an explanation.

        Each result is a dict:
          {
            'item_id': int,
            'title': str,
            'score': float,          # predicted rating
            'contributors': [...]    # which neighbors drove the prediction
          }
        """
        # Guard: unknown user.
        # .index contains all the row labels (user IDs) in the matrix.
        # If this user was not in the training data, we cannot predict anything.
        if user_id not in self.user_item_matrix.index:
            return []

        # .loc[user_id] fetches the row whose label equals user_id.
        # This is label-based lookup — NOT by row number.
        # Result is a Series: index=movie_id, value=rating (NaN if not rated).
        user_row = self.user_item_matrix.loc[user_id]

        # .notna() returns True where a rating exists (i.e. is not NaN).
        # Filtering by that mask keeps only rated movies; .index gives their IDs.
        seen_items = set(user_row[user_row.notna()].index)

        # .columns contains all column labels (movie IDs) in the matrix.
        # Set subtraction removes already-seen movies, leaving only unseen candidates.
        unseen_items = set(self.user_item_matrix.columns) - seen_items

        scored = []
        for item_id in unseen_items:
            predicted = self._predict_single(user_id, item_id)
            if predicted is None:
                continue

            # Look up the movie title; fall back to the numeric id if missing
            title_match = self.items_df[self.items_df["movie_id"] == item_id]["movie_title"]
            title = title_match.values[0] if len(title_match) > 0 else f"Item {item_id}"

            # Collect the neighbors that influenced this prediction the most
            explanation = self._explain_prediction(user_id, item_id)

            scored.append({
                "item_id": item_id,
                "title": title,
                "score": predicted,
                "contributors": explanation,
            })

        # Sort highest predicted rating first and return the top N
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_n]

    # ------------------------------------------------------------------ #
    #  Private helpers — each does exactly one thing                      #
    # ------------------------------------------------------------------ #

    def _build_user_item_matrix(self, ratings_df):
        """Pivot the flat ratings table into a user×item matrix.

        Rows are user IDs, columns are movie IDs, values are ratings (1–5).
        Movies a user has not rated are stored as NaN.
        """
        # .pivot() reshapes a long table (one row per rating) into a wide matrix.
        # index="user_id"   → each unique user becomes a row
        # columns="item_id" → each unique movie becomes a column
        # values="rating"   → the cell value is the rating; NaN where the user hasn't rated
        return ratings_df.pivot(index="user_id", columns="item_id", values="rating")

    def _compute_user_means(self, matrix):
        """Compute each user's average rating across all movies they have rated.

        pandas .mean(axis=1) automatically ignores NaN cells.
        """
        # axis=1 means "go across columns for each row", giving one average per user.
        # (axis=0 would go down rows and give one average per movie instead.)
        # NaN cells (unseen movies) are automatically ignored by pandas .mean().
        return matrix.mean(axis=1)

    def _compute_similarity_matrix(self, matrix, user_means):
        """Compute pairwise cosine similarity between all users using mean-centered ratings.

        Mean-centering: subtract each user's average so a user who always rates 5
        looks the same as one who always rates 3 — only relative preferences matter.
        Means they are now centered around 0.

        Filling NaN with 0 after centering is equivalent to ignoring unrated items
        in the dot product, which is the standard approach for sparse rating data.
        """
        # .subtract(user_means, axis=0) subtracts each user's own average from
        # every rating in that user's row. axis=0 tells pandas to match by row label.
        centered = matrix.subtract(user_means, axis=0)

        # Replace NaN (unseen movies) with 0 so they don't affect dot products
        filled = centered.fillna(0).to_numpy()  # shape: (n_users, n_items)

        # Dot product of all user-vector pairs at once (fast matrix multiplication)
        dot_products = filled @ filled.T          # shape: (n_users, n_users)

        # Compute the length (norm) of each user's rating vector
        norms = np.linalg.norm(filled, axis=1)   # shape: (n_users,)

        # Divide each dot product by the product of the two norms
        # np.errstate suppresses the divide-by-zero warning for users with no ratings
        with np.errstate(invalid="ignore"):
            sim = dot_products / np.outer(norms, norms)

        # Replace any NaN that resulted from 0/0 division with 0
        sim = np.nan_to_num(sim)

        # Wrap the result back in a DataFrame so we can look up by user_id
        users = matrix.index.tolist()
        return pd.DataFrame(sim, index=users, columns=users)

    def _get_k_neighbors(self, user_id, item_id):
        """Find the k most similar users to user_id who have rated item_id.

        Returns a list of (neighbor_id, their_rating, similarity) tuples,
        ordered by similarity (highest first), capped at k entries.
        Only users with positive similarity are included (negative = opposite taste).
        """
        # Guard: unknown user — .index holds all row labels (user IDs)
        if user_id not in self.user_item_matrix.index:
            return []
        # Guard: unknown item — .columns holds all column labels (movie IDs)
        if item_id not in self.user_item_matrix.columns:
            return []

        # Fetch the column for this movie: a Series where index=user_id, value=their rating.
        # Users who haven't rated it have NaN here.
        item_ratings = self.user_item_matrix[item_id]

        # Fetch this user's column from the similarity matrix.
        # The similarity matrix has users as both rows and columns, so this gives
        # a Series: how similar every other user is to our target user.
        user_sims = self.similarity_matrix[user_id]

        neighbors = []
        for neighbor_id, rating in item_ratings.items():
            if neighbor_id == user_id:
                continue              # skip the target user themselves
            if pd.isna(rating):
                continue              # skip users who haven't rated this movie
            sim = user_sims[neighbor_id]
            if sim <= 0:
                continue              # skip users with no positive similarity
            neighbors.append((neighbor_id, rating, float(sim)))

        # Keep only the k most similar neighbors
        neighbors.sort(key=lambda x: x[2], reverse=True)
        return neighbors[: self.k]

    def _predict_single(self, user_id, item_id):
        """Predict the rating user_id would give item_id using KNNWithMeans.

        Formula:
            prediction = user_mean + Σ(sim * (r_neighbor - mean_neighbor)) / Σ(sim)

        Falls back to the user's own average when no neighbors are available.
        Returns None if user_id is unknown.
        """
        # Guard: unknown user or item
        if user_id not in self.user_item_matrix.index:
            return None
        if item_id not in self.user_item_matrix.columns:
            return None

        neighbors = self._get_k_neighbors(user_id, item_id)

        # No neighbors found → use the user's plain average as a fallback
        if not neighbors:
            return float(self.user_means[user_id])

        user_mean = self.user_means[user_id]

        numerator = 0.0    # weighted sum of neighbor deviations from their own mean
        denominator = 0.0  # sum of similarities (used to normalise)

        for neighbor_id, neighbor_rating, sim in neighbors:
            neighbor_mean = self.user_means[neighbor_id]
            # How much did this neighbor deviate from their average? Weight by similarity.
            numerator += sim * (neighbor_rating - neighbor_mean)
            denominator += abs(sim)

        predicted = user_mean + (numerator / denominator)

        # Clamp to the valid rating range [1, 5] — the formula can overshoot
        return float(np.clip(predicted, 1.0, 5.0))

    def _explain_prediction(self, user_id, item_id, top_n=3):
        """Return the top_n neighbors that most influenced the prediction.

        Each entry is a dict:
          {
            'user_id'     : int/str,   # the neighbor's ID
            'similarity'  : float,     # how similar they are to user_id (0-1)
            'rating'      : float,     # what they rated this movie
            'contribution': float,     # positive = pushed prediction up, negative = down
          }
        """
        neighbors = self._get_k_neighbors(user_id, item_id)

        contributors = []
        for neighbor_id, neighbor_rating, sim in neighbors:
            neighbor_mean = self.user_means[neighbor_id]
            # Contribution: how much this neighbor shifted the final predicted rating
            contribution = sim * (neighbor_rating - neighbor_mean)
            contributors.append({
                "user_id": neighbor_id,
                "similarity": round(sim, 3),
                "rating": neighbor_rating,
                "contribution": round(float(contribution), 3),
            })

        # Show the most influential neighbor first
        contributors.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return contributors[:top_n]