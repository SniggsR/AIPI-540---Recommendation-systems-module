
# import package
import sys
import os
import torch
import cornac
import papermill as pm
import scrapbook as sb
import pandas as pd
import pickle as pkl
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED

print("System version: {}".format(sys.version))
print("PyTorch version: {}".format(torch.__version__))
print("Cornac version: {}".format(cornac.__version__))

# read data
df = pd.read_pickle('data/final_dataframe.pkl')

# choose a subset
df_new = df.sample(20000)


# encode data 
data_encoded = df_new.copy()
for col in ['business_id','user_id']:
    data_encoded[col] = data_encoded[col].astype('category') # Convert to category type
    data_encoded[col] = data_encoded[col].cat.codes # Convert to numerical code



# choose the needed columns for training
data_encoded_only = data_encoded[['business_id','user_id','review_stars']]
data_encoded_only = data_encoded_only.reset_index(drop=True)

# rename the columns
data_encoded_only.rename(columns = {'business_id':'itemID', 'user_id':'userID','review_stars':'rating'}, inplace = True)

# split trainging and test set
train, test = python_random_split(data_encoded_only, 0.75)


# set parameters
# top k items to recommend
TOP_K = 10

# Model parameters
LATENT_DIM = 5
ENCODER_DIMS = [300]
ACT_FUNC = "tanh"
LIKELIHOOD = "gaus"
NUM_EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# seend the training data to training set
train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=SEED)

# check the number of users and items
print('Number of users: {}'.format(train_set.num_users))
print('Number of items: {}'.format(train_set.num_items))


# training 
bivae = cornac.models.BiVAECF(
    k=LATENT_DIM,
    encoder_structure=ENCODER_DIMS,
    act_fn=ACT_FUNC,
    likelihood=LIKELIHOOD,
    n_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    seed=SEED,
    use_gpu=torch.cuda.is_available(),
    verbose=True
)

with Timer() as t:
    bivae.fit(train_set)
print("Took {} seconds for training.".format(t))


# generate prediction 

with Timer() as t:
    all_predictions = predict_ranking(bivae, train, usercol='userID', itemcol='itemID', remove_seen=True)
print("Took {} seconds for prediction.".format(t))


eval_map = map_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)

print("MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')


#filename = 'bivae_model.pt'

# Save the entire model
#torch.save(bivae, filename)


#model = torch.load(filename)

#c = predict_ranking(model, train, usercol='userID', itemcol='itemID', remove_seen=True)

