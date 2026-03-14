import numpy as np


def compute_single_rmse(predicted, actual):
    """Compute RMSE for one prediction-target pair.

    Returns None if either value is missing.
    """
    if predicted is None or actual is None:
        return None
    return float(np.sqrt((predicted - actual) ** 2))


def compute_rmse(model, test_data):
    """Compute Root Mean Squared Error (RMSE) for a fitted model on test data.

    For every (user, movie) pair in the test set, ask the model to predict
    the rating and compare it to the actual rating.
    Pairs where the model cannot make a prediction (user or movie was not in
    the training data) are skipped.

    Parameters
    ----------
    model     : a fitted CollaborativeRecommender (or any model with predict_rating)
    test_data : DataFrame with columns [user_id, item_id, rating, ...]

    Returns
    -------
    float : RMSE (lower is better)
    """
    squared_errors = []

    for row in test_data.itertuples():
        predicted = model.predict_rating(row.user_id, row.item_id)

        # Skip if model had no training data for this user or movie
        if predicted is None:
            continue

        actual = row.rating
        squared_errors.append((predicted - actual) ** 2)

    # Mean of squared errors, then square root to get RMSE
    return float(np.sqrt(np.mean(squared_errors)))
