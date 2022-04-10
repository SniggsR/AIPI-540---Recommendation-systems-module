
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#%% md
# Mount Google Drive
#%%
from google.colab import drive
drive.mount('/content/drive')
#%% md
# Import Data
#%%
business_df = pd.read_csv("/content/drive/MyDrive/Corelli/Duke Spring 2022/AIPI 540/Recommendation Systems Module Project/data/business_df_final.csv")
#%%
business_df
#%% md
# Grab 50,000 examples
#%%
business_df = business_df.sample(n=50000)
business_df
#%% md
# Vectorize categories
#%%
# Get vector representations of genre
vec = CountVectorizer()
categories_vec = vec.fit_transform(business_df['categories'])

# Display resulting feature vectors
categories_vectorized = pd.DataFrame(categories_vec.todense(),columns=vec.get_feature_names_out(),index=business_df.business_id)
categories_vectorized.head()

#%% md
# Remove unnecessary categories (top 30 categories)
#%%
csmatrix = cosine_similarity(categories_vec)
csmatrix = pd.DataFrame(csmatrix,columns=business_df.business_id,index=business_df.business_id)
#%%
csmatrix.head(30)
#%%
nc_df = business_df[business_df["state"] == "FL"]
nc_df
#%%
business_df