#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[37]:


business_df = pd.read_json("./business.json", lines=True)


# In[38]:


business_df = business_df[business_df["state"] == "FL"] # grab just businesses in Florida
business_df = business_df.dropna(axis=0,subset=["categories"]) # drop rows with have None in categories (can't do content filtering there obviously)
business_df = business_df.drop(labels=["latitude","longitude","attributes","hours","is_open"], axis=1) 
business_df = business_df.drop_duplicates(subset=['business_id'])
business_df = business_df.reset_index(drop=True)
business_df


# In[193]:


business_df.to_pickle("./final-data/business_df_clean.pkl", protocol=4)


# In[40]:


test_df = pd.DataFrame(csmatrix.index)
test_df


# In[198]:


df


# In[41]:


business_df = pd.merge(business_df, test_df, how="inner", on="business_id")
business_df


# In[200]:


ratings


# In[59]:


new_business_df = pd.merge(business_df, restaurants_series, on="name", how="inner")
new_business_df


# In[57]:


restaurants_series = pd.DataFrame(restaurants_df.index)
restaurants_series


# In[43]:


size = 500000
review = pd.read_json("./review.json", lines=True,
                      dtype={'review_id':str,'user_id':str,
                             'business_id':str,'stars':int,
                             'date':str,'text':str,'useful':int,
                             'funny':int,'cool':int},
                      chunksize=size)


# In[44]:


review


# In[46]:


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


# In[48]:


df = df.drop(labels=["text","date"], axis=1)
df


# In[53]:


df.to_pickle("./final-data/final_df.pkl", protocol=4)


# In[12]:


# There are multiple chunks to be read
chunk_list = []
for chunk_review in review:
    # Drop columns that aren't needed
    #chunk_review = chunk_review.drop(['review_id','useful','funny','cool'], axis=1)
    # Renaming column name to avoid conflict with business overall star rating
    chunk_review = chunk_review.rename(columns={'stars': 'review_stars'})
    # Inner merge with edited business file so only reviews related to the business remain
    chunk_merged = pd.merge(business_df, chunk_review, on='business_id', how='inner')
    # Show feedback on progress
    print(f"{chunk_merged.shape[0]} out of {size:,} related reviews")
    chunk_list.append(chunk_merged)
# After trimming down the review file, concatenate all relevant data back to one dataframe
df = pd.concat(chunk_list, ignore_index=True, join='outer', axis=0)


# In[64]:


df.to_pickle("./final_dataframe.pkl", protocol=4)


# In[14]:


df = df.drop(labels=["useful","funny","cool","text","date"], axis=1)
df


# # Content Filtering

# In[20]:


#business_df = business_df.drop_duplicates(subset=['name'])
business_df


# In[22]:


vec = CountVectorizer()
categories_vec = vec.fit_transform(business_df['categories'])

# Display resulting feature vectors
categories_vectorized = pd.DataFrame(categories_vec.todense(),columns=vec.get_feature_names_out(),index=business_df.business_id)
categories_vectorized


# In[24]:


sums = categories_vectorized.apply(sum, axis =0)
sums.sort_values(ascending=False)


# In[25]:


food_categories= ["restaurants","food"]
categories_vectorized[food_categories]


# In[26]:


food_categories= ["restaurants","food"]
categories_vectorized[food_categories]

restaurants_df = categories_vectorized[categories_vectorized[food_categories].apply(sum, axis=1) >= 1]
restaurants_df


# In[29]:


features_sorted = restaurants_df.apply(sum, axis=0)
features_sorted[features_sorted >= 10]


# In[30]:


features_sorted = restaurants_df.apply(sum, axis=0)
#test_df = restaurants_df[features_sorted >= 10]
#test_df
use_these_features = features_sorted[features_sorted >= 10]

use_these_features


# In[78]:


use_these_features.to_pickle("./FINAL_features.pkl", protocol=4)


# In[31]:


restaurants_df = restaurants_df[use_these_features.index]
restaurants_df


# In[34]:


# Build similarity matrix of movies based on similarity of genres
csmatrix = cosine_similarity(restaurants_df.to_numpy())
csmatrix = pd.DataFrame(csmatrix,columns=restaurants_df.index,index=restaurants_df.index)
csmatrix


# In[35]:


csmatrix.to_pickle("./final-data/csmatrix.pkl", protocol=4)


# In[40]:


csmatrix["Sic Ink"]


# # Adding reviews

# In[113]:


ratings = df[["business_id","user_id","review_stars"]]
ratings


# # Generate Recommendations for user

# In[183]:


def generate_recommendations(user,csmatrix,ratings,business_df):
    # Get top rated restaurant by user
    user_ratings = ratings.loc[ratings['user_id']==user]
    user_ratings = user_ratings.sort_values(by='review_stars',axis=0,ascending=False)
    toprated = user_ratings.iloc[0,:]['business_id']
    # Find most similar restaurants to the user's top rated movie
    sims = csmatrix[toprated]
    mostsimilar = sims.sort_values(ascending=False).index.values
    # Get 10 most similar restaurants excluding the movie itself
    mostsimilar = mostsimilar[1:11]
    # Get titles of restaurants from ids
    toprated_name = business_df[business_df["business_id"]==toprated]["name"].values[0]
    mostsimilar_names = []
    
    for i in range(len(mostsimilar)):
        mostsimilar_names.append(business_df[business_df["business_id"]==mostsimilar[i]]["name"].values[0])
    
    return toprated_name, pd.unique(mostsimilar_names)


# In[184]:


user = "4wMvgdEVpFLCIhFANNBvGA"
toprated, recs = generate_recommendations(user,csmatrix=csmatrix,ratings=ratings,business_df=business_df)
print("User's highest rated place was: {}".format(toprated))


# In[185]:


recs


# In[ ]:





# In[71]:


business_df[business_df["business_id"]=="CcjWb1h0mAoplO_hl2QbLw"]


# In[44]:


user_history_df[user_history_df["user_id"]=="Pi1chY5ZWbQ80tjxQcrn_g"]

