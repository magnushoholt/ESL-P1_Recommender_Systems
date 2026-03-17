import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_utils import (
    create_model_suite,
    ensure_output_dir,
    fit_model,
    load_clean_fold,
    safe_predict_rating,
    timestamp_string,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create boxplots of error distributions for all recommender models."
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Cross-validation folds to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the generated PNG and CSV files. Defaults to outputs/.",
    )
    return parser.parse_args()


def validate_folds(folds):
    invalid = [fold for fold in folds if fold not in {1, 2, 3, 4, 5}]
    if invalid:
        invalid_display = ", ".join(str(fold) for fold in invalid)
        raise ValueError(f"Unsupported fold values: {invalid_display}")


def compute_errors(folds):
    """Compute errors (true - predicted) for all models across selected folds.

    Returns a dict:
      {
        "model_name": [error1, error2, ...],
        ...
      }
    """
    errors_by_model = {model.name: [] for model in create_model_suite()}

    for fold in folds:
        items_df, train_clean, test_clean = load_clean_fold(fold)

        for model in create_model_suite():
            fit_model(model, train_clean, items_df)

            # Compute error for every test prediction
            for row in test_clean.itertuples(index=False):
                predicted = safe_predict_rating(model, row.user_id, row.item_id)
                if predicted is not None:
                    error = row.rating - predicted
                    errors_by_model[model.name].append(error)

    return errors_by_model


def create_boxplot(errors_by_model, output_path):
    """Create a boxplot of error distributions and save as PNG."""
    # Sort models by name for consistent ordering
    model_names = sorted(errors_by_model.keys())
    error_lists = [errors_by_model[name] for name in model_names]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.boxplot(error_lists, labels=model_names)

    ax.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title("Error distribution by recommender model")
    ax.set_ylabel("Error (true rating - predicted rating)")
    ax.set_xlabel("Recommender model")
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_error_table(errors_by_model, output_path):
    """Save error statistics to CSV."""
    # Compute summary statistics
    stats = {}
    for model_name, errors in errors_by_model.items():
        if errors:
            sorted_errors = sorted(errors)
            n = len(errors)
            stats[model_name] = {
                "count": n,
                "mean": sum(errors) / n,
                "std": (sum((x - (sum(errors) / n)) ** 2 for x in errors) / n) ** 0.5,
                "min": min(errors),
                "q25": sorted_errors[int(0.25 * n)],
                "median": sorted_errors[int(0.5 * n)],
                "q75": sorted_errors[int(0.75 * n)],
                "max": max(errors),
            }
        else:
            stats[model_name] = {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "q25": None,
                "median": None,
                "q75": None,
                "max": None,
            }

    stats_df = pd.DataFrame.from_dict(stats, orient="index").sort_index()
    stats_df.index.name = "recommender"
    stats_df.to_csv(output_path)


def main():
    args = parse_args()
    validate_folds(args.folds)

    output_dir = ensure_output_dir(args.output_dir)
    errors_by_model = compute_errors(args.folds)

    timestamp = timestamp_string()
    folds_str = "_".join(str(f) for f in args.folds)
    base_name = f"error_boxplots_folds_{folds_str}_{timestamp}"

    image_path = output_dir / f"{base_name}.png"
    csv_path = output_dir / f"{base_name}.csv"

    create_boxplot(errors_by_model, image_path)
    save_error_table(errors_by_model, csv_path)

    print(f"Saved boxplot to: {image_path}")
    print(f"Saved error statistics to: {csv_path}")


if __name__ == "__main__":
    main()
