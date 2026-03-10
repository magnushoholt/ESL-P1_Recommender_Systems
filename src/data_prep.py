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
    df = pd.read_csv('data/u.item', sep='|',header=None)
    df.columns = ["movie_id", "movie_title", "release_date", "video_release_date",
              "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
              "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film_Noir", "Horror", "Musical", "Mystery", "Romance", "Sci_Fi",
              "Thriller", "War", "Western"]
    return df

def load_info():
    df = pd.read_csv('data/u.info', sep=r' ', header=None)
    return df



# Example usage
if __name__ == "__main__":
    ratings_df = load_data()
    items_df = load_item()
    info_df = load_info()
    print(ratings_df.head())
    print(items_df.head())
    print(info_df.head())
    matrix_df = matrix_data()
    print(matrix_df.head())