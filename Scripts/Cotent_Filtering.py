import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# %%
csmatrix = pd.read_pickle("./csmatrix.pkl")
ratings = pd.read_pickle("./ratings_df.pkl")
business_df = pd.read_pickle("./business_df_clean.pkl")


# %%
# csmatrix --> 10863 business_id x 10863 business_id

# restaurants_df --> 10863 business_ids x 291 food features

# final_df --> 856484 reviews x 11 columns (probably unnecessary)

# business_df_clean --> 10863 business_id x 9 columns

# ratings_df --> 856484 reviews × 3 columns (USE)
# %% md
# Generate Recommendations for User
# %%
def generate_recommendations(user, csmatrix, ratings, business_df):
    # Get top rated restaurant by user
    user_ratings = ratings.loc[ratings['user_id'] == user]
    user_ratings = user_ratings.sort_values(by='review_stars', axis=0, ascending=False)
    toprated = user_ratings.iloc[0, :]['business_id']
    # Find most similar restaurants to the user's top rated movie
    sims = csmatrix[toprated]
    mostsimilar = sims.sort_values(ascending=False).index.values
    # Get 10 most similar restaurants excluding the movie itself
    mostsimilar = mostsimilar[1:11]
    # Get titles of restaurants from ids
    toprated_name = business_df[business_df["business_id"] == toprated]["name"].values[0]

    recommendations_dict = {}

    for i in range(len(mostsimilar)):
        recommendations_dict[business_df[business_df["business_id"] == mostsimilar[i]]["name"].values[0]] = [
            str(business_df[business_df["business_id"] == mostsimilar[i]]["stars"].values[0]) + " stars",
            str(business_df[business_df["business_id"] == mostsimilar[i]]["review_count"].values[0]) + " total reviews"]

    return toprated_name, recommendations_dict


# %%
user = "4wMvgdEVpFLCIhFANNBvGA"
toprated, recs_dict = generate_recommendations(user, csmatrix=csmatrix, ratings=ratings, business_df=business_df)
print("User's highest rated place was: {}".format(toprated))
# %%
recs_dict
# %% md
# Calculate RMSE
# %%
# Split our data into training and validation sets
X = ratings.drop(labels=['review_stars'], axis=1)
y = ratings['review_stars']
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0, test_size=0.2)


# %%
def predict_rating(user_item_pair, simtable=csmatrix, X_train=X_train, y_train=y_train):
    movie_to_rate = user_item_pair['business_id']
    user = user_item_pair['user_id']
    # Filter similarity matrix to only movies already reviewed by user
    movies_watched = X_train.loc[X_train['user_id'] == user, 'business_id'].tolist()
    simtable_filtered = simtable.loc[movie_to_rate, movies_watched]
    # Get the most similar movie already watched to current movie to rate
    try:
        most_similar_watched = simtable_filtered.index[np.argmax(simtable_filtered)]
    except:
        return 3
    # Get user's rating for most similar movie
    idx = X_train.loc[(X_train['user_id'] == user) & (X_train['business_id'] == most_similar_watched)].index.values[0]
    most_similar_rating = y_train.loc[idx]
    return most_similar_rating


# %%
# Get the predicted ratings for each movie in the validation set and calculate the RMSE
ratings_valset = X_val.sample(1000, random_state=0).apply(lambda x: predict_rating(x), axis=1)
val_rmse = np.sqrt(mean_squared_error(y_val.sample(1000, random_state=0), ratings_valset))
print('RMSE of predicted ratings is {:.3f}'.format(val_rmse))