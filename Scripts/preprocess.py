import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data(business_json_path, review_json_path):
    business_df = pd.read_json(business_json_path, lines=True)
    business_df = business_df[business_df["state"] == "FL"] # grab just businesses in Florida
    business_df = business_df.dropna(axis=0,subset=["categories"]) # drop rows with have None in categories (can't do content filtering there obviously)
    business_df = business_df.drop(labels=["latitude","longitude","attributes","hours","is_open"], axis=1)
    business_df = business_df.drop_duplicates(subset=['business_id'])
    business_df = business_df.reset_index(drop=True)

    size = 500000
    review = pd.read_json(review_json_path, lines=True,
                      dtype={'review_id':str,'user_id':str,
                             'business_id':str,'stars':int,
                             'date':str,'text':str,'useful':int,
                             'funny':int,'cool':int},
                      chunksize=size)

    # There are multiple chunks to be read
    chunk_list = []
    for chunk_review in review:
        # Drop columns that aren't needed
        chunk_review = chunk_review.drop(['review_id','useful','funny','cool'], axis=1)
        # Renaming column name to avoid conflict with business overall star rating
        chunk_review = chunk_review.rename(columns={'stars': 'review_stars'})
        # Inner merge with edited business file so only reviews related to the business remain
        chunk_merged = pd.merge(business_df, chunk_review, on='business_id', how='inner')
        # Show feedback on progress
        print(f"{chunk_merged.shape[0]} out of {size:,} related reviews")
        chunk_list.append(chunk_merged)
    # After trimming down the review file, concatenate all relevant data back to one dataframe
    df = pd.concat(chunk_list, ignore_index=True, join='outer', axis=0)

    df = df.drop(labels=["text","date"], axis=1)

    vec = CountVectorizer()
    categories_vec = vec.fit_transform(business_df['categories'])

    # Display resulting feature vectors
    categories_vectorized = pd.DataFrame(categories_vec.todense(),columns=vec.get_feature_names_out(),index=business_df.business_id)
    categories_vectorized

    food_categories= ["restaurants","food"]
    restaurants_df = categories_vectorized[categories_vectorized[food_categories].apply(sum, axis=1) >= 1]

    features_sorted = restaurants_df.apply(sum, axis=0)
    use_these_features = features_sorted[features_sorted >= 10]
    restaurants_df = restaurants_df[use_these_features.index]

    temporary = pd.DataFrame(restaurants_df.index)
    df = pd.merge(temporary, df, on="business_id", how="inner")

    csmatrix = cosine_similarity(restaurants_df.to_numpy())
    csmatrix = pd.DataFrame(csmatrix,columns=restaurants_df.index,index=restaurants_df.index)
    csmatrix # this is csmatrix

    ratings = df[["business_id","user_id","review_stars"]]
    business_df_clean = pd.merge(business_df, temporary, on="business_id", how="inner")

    return csmatrix, restaurants_df, df, business_df_clean, ratings