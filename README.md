# Explainable-statistical-learning
GitHub for Explainable Statistical Learning Project 1: Recommender Systems

## How to create a virtual environment and install dependencies
1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Generate a per-user prediction plot
This script plots the ratings present in one test fold for a single user and saves both a PNG plot and a CSV with the underlying values in `outputs/`.

```bash
python src/plot_user_test_predictions.py --user-id 1 --fold 1
```

Optional arguments let you choose the collaborative configuration and content-model rating bias:

```bash
python src/plot_user_test_predictions.py --user-id 1 --fold 3 --collab-k 25 --collab-similarity cosine --content-include-rating --content-rating-bias 10
```

## Export an RMSE table for all recommenders
This script runs the same recommender suite as `src/main.py` across the selected folds and saves the RMSE table to `outputs/` as CSV and TXT.

```bash
python src/export_rmse_table.py
```

You can also restrict the evaluated folds:

```bash
python src/export_rmse_table.py --folds 1 2 3
```