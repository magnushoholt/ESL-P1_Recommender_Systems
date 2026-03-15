import numpy as np
import pandas as pd
from base_recommender import BaseRecommender
from data_prep import matrix_data


class CollaborativeRecommender(BaseRecommender):
    """Collaborative filtering recommender with explainable predictions.

    How it works:
      1. fit()           — build a user x item rating matrix and precompute similarities.
      2. predict_rating()— estimate what rating a user would give an unseen movie.
      3. recommend()     — return the top-N unseen movies with an explanation for each.

    The prediction formula (KNNWithMeans-style) is:
        predicted = user_mean + Σ(sim * (neighbor_rating - neighbor_mean)) / Σ(sim)

    This class supports both:
      - user-user collaborative filtering (user_based=True)
      - item-item collaborative filtering (user_based=False)

    Mean-centering removes bias from users/items that always rate high or low.
    """

    def __init__(self, name, k=3, user_based=True, similarity="cosine"):
        """Store model configuration.

        Parameters
        ----------
        name : str
            A label for this model instance, e.g. "Collaborative".
            Useful when comparing multiple models side by side.
        k : int, optional (default=3)
            How many similar users to consult when predicting a rating.
        user_based : bool, optional (default=True)
            True  -> user-user collaborative filtering.
            False -> item-item collaborative filtering.
        similarity : str, optional (default="cosine")
            Similarity metric. Supported: "cosine", "pearson".
        """
        super().__init__(name)
        self.k = k
        self.user_based = user_based
        self.similarity = similarity
        self.user_item_matrix = None   # rows=users, cols=items, values=ratings (NaN if unseen)
        self.user_means = None         # average rating per user
        self.item_means = None         # average rating per item
        self.similarity_matrix = None  # user-user or item-item similarity matrix
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

        # Compute means used in KNNWithMeans-style prediction.
        # We keep both so user-based and item-based modes can share code paths.
        self.user_means = self._compute_user_means(self.user_item_matrix)
        self.item_means = self.user_item_matrix.mean(axis=0)

        # Precompute either user-user or item-item similarity matrix.
        self.similarity_matrix = self._compute_similarity_matrix(self.user_item_matrix)

    def predict_rating(self, user_id, item_id, display=False):
        """Return the predicted rating (1-5) that user_id would give item_id.

        Returns None if the user or item is not in the training data.
        """
        score = self._predict_single(user_id, item_id)
        if display:
            self.display_prediction(
                user_id=user_id,
                item_id=item_id,
                predicted_score=score,
            )
        return score

    def display_prediction(self, user_id, item_id, predicted_score):
        """Print one prediction for a user-item pair."""
        title = self._movie_title(item_id)

        if predicted_score is None:
            print(f"Predicted rating for user {user_id}, film: '{title}' - unavailable")
        else:
            print(f"Predicted rating for user {user_id}, film: '{title}' - {predicted_score:.1f}")

    def display_recommendations(self, user_id, recommendations, top_n):
        """Print top-N recommendation output with neighbor details."""
        print(f"Top {top_n} recommendations for user {user_id}:")
        if not recommendations:
            print("  No recommendations available.")
            return

        for rec in recommendations:
            print(f"  {rec['title']} (predicted: {rec['score']:.1f})")
            source_label = "user" if self.user_based else "item"
            for c in rec["contributors"]:
                print(
                    f"    neighbor {source_label} {c['source_id']} "
                    f"sim={c['similarity']:.5f} "
                    f"rated {c['rating']:.1f} "
                    f"contrib={c['contribution']:+.3f}"
                )

    def recommend(self, user_id, top_n=5, display=False):
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
            title = self._movie_title(item_id)

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
        out = scored[:top_n]

        if display:
            self.display_recommendations(user_id=user_id, recommendations=out, top_n=top_n)

        return out

    # ------------------------------------------------------------------ #
    #  Private helpers — each does exactly one thing                      #
    # ------------------------------------------------------------------ #

    def _build_user_item_matrix(self, ratings_df):
        """Pivot the flat ratings table into a user×item matrix.

        Rows are user IDs, columns are movie IDs, values are ratings (1–5).
        Movies a user has not rated are stored as NaN.
        """
        
        return matrix_data(ratings_df)

    def _movie_title(self, item_id):
        """Resolve a movie title from item_id, with a fallback label."""
        title_match = self.items_df[self.items_df["movie_id"] == item_id]["movie_title"]
        return title_match.values[0] if len(title_match) > 0 else f"Item {item_id}"

    def _compute_user_means(self, matrix):
        """Compute each user's average rating across all movies they have rated.

        pandas .mean(axis=1) automatically ignores NaN cells.
        """
        # axis=1 means "go across columns for each row", giving one average per user.
        # (axis=0 would go down rows and give one average per movie instead.)
        # NaN cells (unseen movies) are automatically ignored by pandas .mean().
        return matrix.mean(axis=1)

    def _compute_similarity_matrix(self, matrix):
        """Compute user-user or item-item similarities from centered rating vectors.

        High-level:
          - user-based mode compares users (rows)
          - item-based mode compares items (columns)

        Low-level:
          - center vectors by user/item mean
          - fill NaN with 0 for sparse operations
          - compute pairwise similarity by selected metric
        """
        if self.user_based:
            centered = matrix.subtract(self.user_means, axis=0)
            vectors = centered.fillna(0).to_numpy()  # shape: (n_users, n_items)
            labels = matrix.index.tolist()
        else:
            centered = matrix.subtract(self.item_means, axis=1)
            vectors = centered.fillna(0).to_numpy().T  # shape: (n_items, n_users)
            labels = matrix.columns.tolist()

        sim = self._pairwise_similarity(vectors)
        return pd.DataFrame(sim, index=labels, columns=labels)

    def _pairwise_similarity(self, vectors):
        """Compute pairwise similarity for a matrix of entity vectors.

        vectors shape: (n_entities, n_features)
        """
        if self.similarity == "cosine":
            dot_products = vectors @ vectors.T
            norms = np.linalg.norm(vectors, axis=1)
            with np.errstate(invalid="ignore"):
                sim = dot_products / np.outer(norms, norms)
            return np.nan_to_num(sim)

        if self.similarity == "pearson":
            # Pearson across entities (rows). NaN appears for constant vectors, so map to 0.
            return np.nan_to_num(np.corrcoef(vectors))

        raise ValueError(f"Unsupported similarity: {self.similarity}")

    def _get_k_neighbors(self, user_id, item_id):
        """Return top-k neighbors for the current collaborative mode.

        user-based:
          neighbors are similar users who rated target item.
        item-based:
          neighbors are similar items already rated by target user.
        """
        if user_id not in self.user_item_matrix.index or item_id not in self.user_item_matrix.columns:
            return []

        neighbors = []

        if self.user_based:
            item_ratings = self.user_item_matrix[item_id]
            user_sims = self.similarity_matrix[user_id]
            for neighbor_user_id, rating in item_ratings.items():
                if neighbor_user_id == user_id or pd.isna(rating):
                    continue
                sim = user_sims[neighbor_user_id]
                if sim <= 0:
                    continue
                neighbors.append((neighbor_user_id, rating, float(sim)))
        else:
            user_ratings = self.user_item_matrix.loc[user_id]
            item_sims = self.similarity_matrix[item_id]
            for neighbor_item_id, rating in user_ratings.items():
                if neighbor_item_id == item_id or pd.isna(rating):
                    continue
                sim = item_sims[neighbor_item_id]
                if sim <= 0:
                    continue
                neighbors.append((neighbor_item_id, rating, float(sim)))

        neighbors.sort(key=lambda x: x[2], reverse=True)
        return neighbors[: self.k]

    def _predict_single(self, user_id, item_id):
        """Predict one rating using user-user or item-item KNNWithMeans-style logic.

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

        if self.user_based:
            baseline = self.user_means[user_id]
        else:
            baseline = self.item_means[item_id]

        # No neighbors found -> fall back to baseline mean.
        if not neighbors:
            return float(baseline)

        numerator = 0.0
        denominator = 0.0

        for neighbor_id, neighbor_rating, sim in neighbors:
            # In user-based mode, neighbor mean is a user mean.
            # In item-based mode, neighbor mean is an item mean.
            neighbor_mean = self.user_means[neighbor_id] if self.user_based else self.item_means[neighbor_id]
            numerator += sim * (neighbor_rating - neighbor_mean)
            denominator += abs(sim)

        predicted = baseline + (numerator / denominator)

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
            neighbor_mean = self.user_means[neighbor_id] if self.user_based else self.item_means[neighbor_id]
            # Contribution: how much this neighbor shifted the final predicted rating
            contribution = sim * (neighbor_rating - neighbor_mean)
            contributors.append({
                # Generic source_id works for both user-neighbors and item-neighbors.
                "source_id": neighbor_id,
                "similarity": round(sim, 3),
                "rating": neighbor_rating,
                "contribution": round(float(contribution), 3),
            })

        # Show the most influential neighbor first
        contributors.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return contributors[:top_n]