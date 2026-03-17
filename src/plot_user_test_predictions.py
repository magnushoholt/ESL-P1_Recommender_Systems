import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from collaborative import CollaborativeRecommender
from content import Content_recommender_system
from experiment_utils import (
    ensure_output_dir,
    fit_model,
    load_clean_fold,
    movie_title_from_id,
    safe_predict_rating,
    timestamp_string,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot true test ratings and two model predictions for a single user."
    )
    parser.add_argument("--user-id", type=int, required=True, help="Target user id to plot.")
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 6), help="Cross-validation fold to evaluate.")
    parser.add_argument("--collab-k", type=int, default=10, help="Number of neighbors for the collaborative model.")
    parser.add_argument(
        "--collab-similarity",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Similarity metric for the collaborative model.",
    )
    parser.add_argument(
        "--item-based",
        action="store_true",
        help="Use item-item collaborative filtering instead of user-user.",
    )
    parser.add_argument(
        "--content-include-rating",
        action="store_true",
        help="Include item rating bias as an extra content feature.",
    )
    parser.add_argument(
        "--content-rating-bias",
        type=int,
        default=5,
        help="Bias term used when computing the optional rating feature.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the generated PNG and CSV files. Defaults to outputs/.",
    )
    return parser.parse_args()


def build_models(args):
    collab_prefix = "Item-Item" if args.item_based else "User-User"
    collab_name = f"{collab_prefix} {args.collab_similarity.title()} k={args.collab_k}"
    content_name = "Content (Genres Only)"
    if args.content_include_rating:
        content_name = f"Content (Genres + Rating bias={args.content_rating_bias})"

    collab_model = CollaborativeRecommender(
        collab_name,
        k=args.collab_k,
        user_based=not args.item_based,
        similarity=args.collab_similarity,
    )
    content_model = Content_recommender_system(content_name)
    return collab_model, content_model


def build_user_prediction_frame(user_id, test_data, items_df, collab_model, content_model):
    user_test = test_data[test_data["user_id"] == user_id].sort_values("item_id")
    if user_test.empty:
        raise ValueError(f"User {user_id} has no ratings in the selected test fold.")

    rows = []
    for row in user_test.itertuples(index=False):
        rows.append(
            {
                "item_id": int(row.item_id),
                "movie_title": movie_title_from_id(items_df, row.item_id),
                "true_rating": float(row.rating),
                "collaborative_prediction": safe_predict_rating(collab_model, row.user_id, row.item_id),
                "content_prediction": safe_predict_rating(content_model, row.user_id, row.item_id),
            }
        )

    return pd.DataFrame(rows)


def create_plot(plot_df, user_id, fold, collab_label, content_label, output_path):
    positions = range(len(plot_df))
    figure_width = max(10, len(plot_df) * 0.55)
    fig, ax = plt.subplots(figsize=(figure_width, 6))

    ax.plot(positions, plot_df["true_rating"], marker="o", linewidth=2, label="True rating")
    ax.plot(
        positions,
        plot_df["collaborative_prediction"],
        marker="s",
        linewidth=2,
        label=collab_label,
    )
    ax.plot(
        positions,
        plot_df["content_prediction"],
        marker="^",
        linewidth=2,
        label=content_label,
    )

    ax.set_title(f"Test-set ratings for user {user_id} in fold {fold}")
    ax.set_xlabel("Item id")
    ax.set_ylabel("Rating / score")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(plot_df["item_id"].astype(str), rotation=60, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.text(
        0.01,
        0.01,
        "Content predictions come from the current cosine-similarity content model score.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    items_df, train_clean, test_clean = load_clean_fold(args.fold)

    collab_model, content_model = build_models(args)
    fit_model(collab_model, train_clean, items_df)
    fit_model(content_model, train_clean, items_df)

    plot_df = build_user_prediction_frame(
        user_id=args.user_id,
        test_data=test_clean,
        items_df=items_df,
        collab_model=collab_model,
        content_model=content_model,
    )

    timestamp = timestamp_string()
    base_name = f"user_{args.user_id}_fold_{args.fold}_{timestamp}"
    csv_path = output_dir / f"{base_name}.csv"
    image_path = output_dir / f"{base_name}.png"

    plot_df.to_csv(csv_path, index=False)
    create_plot(
        plot_df=plot_df,
        user_id=args.user_id,
        fold=args.fold,
        collab_label=collab_model.name,
        content_label=content_model.name,
        output_path=image_path,
    )

    print(f"Saved plot data to: {csv_path}")
    print(f"Saved plot image to: {image_path}")


if __name__ == "__main__":
    main()