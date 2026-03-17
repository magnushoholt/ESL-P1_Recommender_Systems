from datetime import datetime
from pathlib import Path
import re

from collaborative import CollaborativeRecommender
from content import Content_recommender_system
from data_prep import cleaning_data, load_cv_split, load_item


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def timestamp_string():
    """Return a filesystem-safe timestamp for output filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_output_dir(output_dir=None):
    """Create and return the directory used for generated artifacts."""
    resolved = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_clean_fold(split_number):
    """Load one MovieLens split and apply the project's cleaning routine."""
    items_df = load_item()
    train_raw, test_raw = load_cv_split(split_number)

    _, train_clean = cleaning_data(items_df.copy(), train_raw.copy())
    _, test_clean = cleaning_data(items_df.copy(), test_raw.copy())
    return items_df, train_clean, test_clean


def movie_title_from_id(items_df, item_id):
    """Resolve a movie title from its id, falling back to a generic label."""
    title_match = items_df[items_df["movie_id"] == item_id]["movie_title"]
    if title_match.empty:
        return f"Item {item_id}"
    return title_match.iloc[0]


def to_scalar_score(value):
    """Convert scalar-or-list score values into a plain float."""
    while isinstance(value, (list, tuple)) and len(value) > 0:
        value = value[0]
    return float(value)


def safe_predict_rating(model, user_id, item_id):
    """Predict one score and return None when the model cannot score the pair."""
    try:
        score = model.predict_rating(user_id=user_id, item_id=item_id)
    except (KeyError, IndexError, ValueError):
        return None

    if score is None:
        return None

    return to_scalar_score(score)


def fit_model(model, train_data, items_df):
    """Fit one model using the same conventions as the main comparison script."""
    if isinstance(model, Content_recommender_system):
        include_rating = "Rating" in model.name
        bias_match = re.search(r"bias=(\d+)", model.name)
        rating_bias = int(bias_match.group(1)) if bias_match else 0
        model.fit(train_data, items_df, include_rating=include_rating, rating_bias=rating_bias)
    else:
        model.fit(train_data, items_df)
    return model


def create_model_suite():
    """Return the model variants used in the project comparison runs."""
    return [
        CollaborativeRecommender("User-User Cosine k=3", k=3, user_based=True, similarity="cosine"),
        CollaborativeRecommender("User-User Cosine k=10", k=10, user_based=True, similarity="cosine"),
        CollaborativeRecommender("User-User Cosine k=25", k=25, user_based=True, similarity="cosine"),
        CollaborativeRecommender("Item-Item Cosine k=3", k=3, user_based=False, similarity="cosine"),
        CollaborativeRecommender("User-User Euclidean k=3", k=3, user_based=True, similarity="euclidean"),
        Content_recommender_system("Content (Genres Only)"),
        Content_recommender_system("Content (Genres + Rating bias=0)"),
        Content_recommender_system("Content (Genres + Rating bias=1)"),
        Content_recommender_system("Content (Genres + Rating bias=5)"),
    ]