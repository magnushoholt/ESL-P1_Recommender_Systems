#!/usr/bin/env python3
"""
Interactive film-recommendation script using the MovieLens 100K dataset.

Workflow:
  1. Display a welcome banner.
  2. Show random movie samples as inspiration.
  3. Ask the user to pick a user ID to "roleplay as" (avoids cold-start).
  4. Prompt for a movie title, validate with fuzzy matching, and let the user
     confirm or pick from the top-3 closest matches.
  5. Train collaborative and content-based recommenders on the full dataset
     and show recommendations for the chosen user.
"""

import os
import random
import difflib
import sys
import time
import argparse

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — ensure src/ is on sys.path and we're in the project root
# so that data_prep's relative paths (data/u.data etc.) resolve correctly.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from data_prep import load_data, load_item, cleaning_data
from collaborative import CollaborativeRecommender
from content import Content_recommender_system

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FAST_MODE = False  # set by parse_args()

WELCOME = (
    "\n"
    "╔═════════════════════════════╗\n"
    "║   Recommender Systems       ║\n"
    "║                             ║\n"  
    "║   Made with help from:      ║\n"
    "║   Qwopus3.6-27B Coder-MTP   ║\n"
    "╚═════════════════════════════╝\n"
)

RANDOM_SAMPLE_COUNT = 3
TOP_N_RECS = 3
FUZZY_TOP_N = 5
PAUSE_SECONDS = 1

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive film-recommendation script using the MovieLens 100K dataset."
    )
    parser.add_argument(
        "-fast",
        action="store_true",
        default=False,
        help="Skip all pause delays for faster execution.",
    )
    return parser.parse_args()


def pause(seconds=PAUSE_SECONDS):
    """Pause execution for *seconds* and print a brief indicator.
    Skipped entirely when _FAST_MODE is True."""
    if _FAST_MODE:
        return
    #print(f"   ⏸  Pausing for {seconds} second(s)…")
    time.sleep(seconds)


def _extract_genres(row, items_df):
    """Return a list of genre names for a single movie row."""
    return [col for col in items_df.columns
            if col.startswith(("Action", "Adventure", "Animation", "Children",
                              "Comedy", "Crime", "Documentary", "Drama",
                              "Fantasy", "Film_Noir", "Horror", "Musical",
                              "Mystery", "Romance", "Sci_Fi", "Thriller",
                              "War", "Western"))
            and row[col] == 1]


def show_random_movies(items_df, count=RANDOM_SAMPLE_COUNT):
    """Print *count* random movies from the catalog as inspiration,
    with vertically aligned genre columns."""
    sample = items_df.sample(n=min(count, len(items_df)), random_state=random.randint(0, 10000))
    print(f"\n🎬 Here are {count} random films from the MovieLens catalog for inspiration:\n")
    pause()
    # Find the longest title in this sample so we can pad uniformly
    max_title_len = max(len(row["movie_title"]) for _, row in sample.iterrows())

    for _, row in sample.iterrows():
        title = row["movie_title"]
        genres = _extract_genres(row, items_df)
        genre_str = ", ".join(genres) if genres else "No genre"
        # Pad the title so all genre brackets line up
        print(f"  • {title:<{max_title_len}}  [{genre_str}]")
        pause()
    print()


def pick_user_id(ratings_df):
    """Prompt the user to select a valid user ID from the dataset."""
    valid_ids = sorted(ratings_df["user_id"].unique())
    print(f"👤 There are {len(valid_ids)} users in the dataset (IDs: {min(valid_ids)}–{max(valid_ids)}).")
    pause()
    print("   Picking a user gives the models real data to work with.\n")
    pause(2)

    while True:
        raw = input("Enter a user ID to roleplay as (or press Enter for a random user): ").strip()
        if raw == "":
            uid = random.choice(valid_ids)
            print(f"   → Randomly selected user {uid}.")
            return uid
        try:
            uid = int(raw)
        except ValueError:
            print("   ⚠  Please enter a valid integer.")
            continue
        if uid not in valid_ids:
            print(f"   ⚠  User {uid} not found. Valid range: {min(valid_ids)}–{max(valid_ids)}.")
            continue
        print(f"   → You are now user {uid}.")
        return uid


def show_user_top_films(ratings_df, items_df, user_id, top_n=3):
    """Print the top *top_n* films for *user_id* by rating, with their scores."""
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]
    top = user_ratings.nlargest(top_n, "rating")
    print(f"\n🏆 Top {top_n} films for user {user_id}:")
    # Resolve titles first so we can compute max length
    titles = []
    for _, row in top.iterrows():
        title_row = items_df[items_df["movie_id"] == row["item_id"]]
        title = title_row["movie_title"].iloc[0] if not title_row.empty else f"Unknown (ID {row['item_id']})"
        titles.append((title, row["rating"]))
    max_title_len = max(len(t) for t, _ in titles)
    for title, rating in titles:
        print(f"   ★ {title:<{max_title_len}}  (rating: {rating:.1f} / 5)")
        pause()


def fuzzy_match_title(query, items_df, top_n=FUZZY_TOP_N):
    """Return the top *top_n* movie titles most similar to *query* (case-insensitive)."""
    all_titles = items_df["movie_title"].tolist()
    ratios = [difflib.SequenceMatcher(None, query.lower(), t.lower()).ratio() for t in all_titles]
    # Build list of (ratio, title) and sort descending
    scored = sorted(zip(ratios, all_titles), key=lambda x: x[0], reverse=True)
    return scored[:top_n]


def resolve_movie_title(items_df):
    """
    Prompt the user for a movie title.
    - If an exact match is found, return the (movie_id, title).
    - If the user presses Enter, pick a random movie from the catalog.
    - Otherwise, show the top-N fuzzy matches and let the user confirm or pick.
    """
    while True:
        query = input("\n🎥 Enter a movie title (or press Enter for a random pick): ").strip()
        if not query:
            row = items_df.sample(n=1, random_state=random.randint(0, 10000)).iloc[0]
            print(f"   🎲 Randomly selected: \"{row['movie_title']}\" (ID {row['movie_id']})")
            return row["movie_id"], row["movie_title"]

        # Exact match (case-insensitive)
        exact = items_df[items_df["movie_title"].str.lower() == query.lower()]
        if not exact.empty:
            row = exact.iloc[0]
            print(f"   ✓ Found: \"{row['movie_title']}\" (ID {row['movie_id']})")
            return row["movie_id"], row["movie_title"]

        # Fuzzy match
        matches = fuzzy_match_title(query, items_df, top_n=FUZZY_TOP_N)
        print(f"\n   ⚠  No exact match for \"{query}\". Top {FUZZY_TOP_N} similar titles:\n")
        for i, (ratio, title) in enumerate(matches, 1):
            print(f"      {i}. {title}  (similarity: {ratio:.2f})")

        # Prompt for confirmation
        while True:
            choice = input("\n   Press Enter to accept #1, or type 1/2/3 to choose: ").strip()
            if choice == "":
                idx = 0
                break
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(matches):
                    break
                print("   ⚠  Please enter 1, 2, or 3.")
            except ValueError:
                print("   ⚠  Invalid input. Press Enter or type 1/2/3.")

        ratio, title = matches[idx]
        movie_id = items_df[items_df["movie_title"] == title].iloc[0]["movie_id"]
        print(f"   ✓ Selected: \"{title}\" (ID {movie_id})")
        return movie_id, title


def _to_scalar_score(value):
    """Convert scalar-or-list score values into a printable float."""
    while isinstance(value, (list, tuple)) and len(value) > 0:
        value = value[0]
    return float(value)


def display_collab_recs(model, user_id, top_n):
    """Display recommendations from a CollaborativeRecommender."""
    recs = model.recommend(user_id=user_id, top_n=top_n, display=False)
    if not recs:
        print("   No recommendations available from this model.")
        return
    for rec in recs:
        print(f"   ★ {rec['title']}  (predicted: {rec['score']:.1f})")
        source_label = "user" if model.user_based else "item"
        for c in rec["contributors"][:2]:  # show at most 2 neighbors
            print(
                f"      neighbor {source_label} {c['source_id']} "
                f"sim={c['similarity']:.4f} rated {c['rating']:.1f} "
                f"contrib={c['contribution']:+.3f}"
            )


def display_content_recs(model, user_id, items_df, top_n):
    """Display recommendations from a Content_recommender_system."""
    recs_df = model.recommend(user_id=user_id, top_n=top_n)
    if recs_df is None or recs_df.empty:
        print("   No recommendations available from this model.")
        return
    for movie_id, row in recs_df.iterrows():
        score = _to_scalar_score(row["Score"])
        title = row["movie_title"]
        print(f"   ★ {title}  (score: {score:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------




def main():
    global _FAST_MODE
    args = parse_args()
    _FAST_MODE = args.fast

    print(WELCOME)
    pause(2)

    # --- Load data --------------------------------------------------------
    print("Loading MovieLens 100K dataset …")
    pause()
    raw_ratings = load_data()
    raw_items = load_item()
    items_clean, ratings_clean = cleaning_data(raw_items, raw_ratings)
    print(f"   {len(items_clean)} movies, {len(ratings_clean)} ratings loaded.")
    pause()

    # --- Show random samples -----------------------------------------------
    show_random_movies(items_clean)
    pause()

    # --- Pick a user -------------------------------------------------------
    user_id = pick_user_id(ratings_clean)
    pause()

    # --- Show user's top films ---------------------------------------------
    show_user_top_films(ratings_clean, items_clean, user_id, top_n=3)
    pause()

    # --- Resolve a movie title ---------------------------------------------
    movie_id, movie_title = resolve_movie_title(items_clean)
    pause()

    # --- Train models ------------------------------------------------------
    print(f"\n🔧 Training recommender models for user {user_id} …")


    #print("\n🔧 Training Collaborative: user-user cosine k=10 …")
    # Collaborative: user-user cosine k=10
    collab_model = CollaborativeRecommender(
        name="User-User Cosine k=10",
        k=10,
        user_based=True,
        similarity="cosine",
    )
    collab_model.fit(ratings_clean, items_clean)

    #print("\n🔧 Training Collaborative: item-item cosine k=10 …")
    # Collaborative: item-item cosine k=10
    collab_item = CollaborativeRecommender(
        name="Item-Item Cosine k=10",
        k=10,
        user_based=False,
        similarity="cosine",
    )
    collab_item.fit(ratings_clean, items_clean)

    #print("\n🔧 Training Content-based: genres only …")
    # Content-based: genres only
    content_model = Content_recommender_system(name="Content (Genres Only)")
    content_model.fit(ratings_clean, items_clean, include_rating=False)

    #print("\n🔧 Training Content-based: genres + rating bias …")
    # Content-based: genres + rating bias
    content_model_r = Content_recommender_system(name="Content (Genres + Rating)")
    content_model_r.fit(ratings_clean, items_clean, include_rating=True, rating_bias=5)
    pause()

    # --- Predict rating for the chosen film --------------------------------
    print(f"\n{'='*70}")
    print(f"  Predicted rating of \"{movie_title}\" for user {user_id}")
    print(f"{'='*70}")

    # Collect all predictions first so we can align the output
    predictions = []
    for model in [collab_model, collab_item]:
        pred = model.predict_rating(user_id, movie_id)
        label = "Collab (User-User)" if model.user_based else "Collab (Item-Item)"
        predictions.append((label, pred))

    pred_c = content_model.predict_rating(user_id, movie_id)
    pred_cr = content_model_r.predict_rating(user_id, movie_id)
    predictions.append(("Content (genres)", pred_c))
    predictions.append(("Content (genres+rating)", pred_cr))

    # Find the longest label so we can pad uniformly
    max_label_len = max(len(label) for label, _ in predictions)

    for label, pred in predictions:
        if pred is not None:
            print(f"   {label:<{max_label_len}} → {pred:.1f} / 5")
        else:
            print(f"   {label:<{max_label_len}} → unavailable")
        pause()
    pause(1)

    # --- Recommendations ---------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  Top {TOP_N_RECS} Recommendations for user {user_id}")
    print(f"{'='*55}")
    pause(1)

    print(f"\n  📊 User-User Collaborative (k=10, cosine):")
    display_collab_recs(collab_model, user_id, TOP_N_RECS)
    pause(1)

    print(f"\n  📊 Item-Item Collaborative (k=10, cosine):")
    display_collab_recs(collab_item, user_id, TOP_N_RECS)
    pause(1)

    print(f"\n  📊 Content-Based (Genres Only):")
    display_content_recs(content_model, user_id, items_clean, TOP_N_RECS)
    pause(1)

    print(f"\n  📊 Content-Based (Genres + Rating Bias):")
    display_content_recs(content_model_r, user_id, items_clean, TOP_N_RECS)
    pause(1)

    print(" ")
    print("  Done.")
    #print(f"{'='*55}\n")


if __name__ == "__main__":
    main()

