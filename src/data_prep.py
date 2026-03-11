import pandas as pd


def load_data():
   df = pd.read_csv('data/u.data', sep='\t', header=None)
   df.columns = ["user_id", "item_id", "rating", "timestamp"]
   return df

def matrix_data():
    df = load_data()
    matrix_df = df.pivot(index='user_id', columns='item_id', values='rating')
    return matrix_df


def load_item():
    df = pd.read_csv('data/u.item', sep='|', header=None, encoding='latin-1')
    df.columns = ["movie_id", "movie_title", "release_date", "video_release_date",
              "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
              "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film_Noir", "Horror", "Musical", "Mystery", "Romance", "Sci_Fi",
              "Thriller", "War", "Western"]
    return df

def load_info():
    df = pd.read_csv('data/u.info', sep=r' ', header=None)
    return df

def cleaning_data(items_df, ratings_df):
    # Find duplicates
    dup = items_df.groupby("movie_title")["movie_id"].apply(list)
    duplicates = dup[dup.str.len() > 1]
    
    # Mapping
    mapping = {ids[1]: ids[0] for ids in duplicates}
    ids_to_change = list(mapping.keys())

    # For testing if everything is correct after
    untouched_mask = ~ratings_df["item_id"].isin(ids_to_change)
    
    # Saving data not to change
    expected_untouched_data = ratings_df[untouched_mask].copy()

    # Change values
    ratings_df["item_id"] = ratings_df["item_id"].replace(mapping)

    # Check if all is changed (should be 0)
    remaining_duplicates = ratings_df["item_id"].isin(ids_to_change).sum()
    
    # Check if all other data is intact
    current_untouched_data = ratings_df[untouched_mask]
    integrity_check = expected_untouched_data.equals(current_untouched_data)
    
    # Print Results
    if not integrity_check:
        print(f"1. Duplicate IDs remaining: {remaining_duplicates} (Expected: 0)")
        print(f"2. Data Integrity Check: FAILED")

    # Clean data and preserve order
    rating_column_order = ratings_df.columns.tolist()
    ratings_clean = ratings_df.groupby(["user_id","item_id"], as_index=False).mean() #Fucker tiden up men virker for alt andet
    ratings_clean = ratings_clean.reindex(columns=rating_column_order)

    item_column_order = items_df.columns.tolist()
    items_clean = items_df.groupby(["movie_title"], as_index=False).min()
    items_clean = items_clean.reindex(columns=item_column_order)

    return items_clean, ratings_clean


# Example usage
if __name__ == "__main__":
    ratings_df = load_data()
    items_df = load_item()
    print(f"Number of movies with multiple id's: {items_df["movie_title"].value_counts().value_counts()}")
    item_clean, ratings_clean = cleaning_data(items_df,ratings_df)
    print(ratings_clean.head())
    print(f"Number of ratings after cleaning: {ratings_clean.shape[0]}")
    print(item_clean.head())
    info_df = load_info()
    print(info_df.head())
    matrix_df = matrix_data()
    print(matrix_df.head())