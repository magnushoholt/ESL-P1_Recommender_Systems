import argparse

import pandas as pd

from eval import compute_rmse
from experiment_utils import create_model_suite, ensure_output_dir, fit_model, load_clean_fold, timestamp_string


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the recommender suite across folds and save the RMSE table."
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
        default=None,
        help="Directory for generated RMSE tables. Defaults to outputs/.",
    )
    return parser.parse_args()


def validate_folds(folds):
    invalid = [fold for fold in folds if fold not in {1, 2, 3, 4, 5}]
    if invalid:
        invalid_display = ", ".join(str(fold) for fold in invalid)
        raise ValueError(f"Unsupported fold values: {invalid_display}")


def compute_rmse_table(folds):
    rmse_by_model = {}

    for fold in folds:
        items_df, train_clean, test_clean = load_clean_fold(fold)

        for model in create_model_suite():
            fit_model(model, train_clean, items_df)
            rmse = compute_rmse(model, test_clean)
            rmse_by_model.setdefault(model.name, {})[f"fold_{fold}"] = rmse

    result = pd.DataFrame.from_dict(rmse_by_model, orient="index")
    ordered_columns = [f"fold_{fold}" for fold in folds]
    result = result.reindex(columns=ordered_columns)
    result.index.name = "recommender"
    return result.sort_index()


def main():
    args = parse_args()
    validate_folds(args.folds)

    output_dir = ensure_output_dir(args.output_dir)
    rmse_table = compute_rmse_table(args.folds)

    timestamp = timestamp_string()
    csv_path = output_dir / f"rmse_table_{timestamp}.csv"
    txt_path = output_dir / f"rmse_table_{timestamp}.txt"

    rmse_table.to_csv(csv_path, float_format="%.4f")
    txt_path.write_text(rmse_table.to_string(float_format=lambda value: f"{value:.4f}"), encoding="utf-8")

    print(f"Saved RMSE CSV to: {csv_path}")
    print(f"Saved RMSE text table to: {txt_path}")


if __name__ == "__main__":
    main()