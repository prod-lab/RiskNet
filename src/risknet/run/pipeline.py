'''
This is where everything runs!
We call various functions from .model, .reducer, .encoder, etc. step-by-step to run the pipeline.
Check out the comments to see what each part of the code does.
'''

#Global Imports:
import pandas as pd
from pandas import DataFrame

import numpy as np

from typing import List, Dict, Tuple
import pickle
import logging
logger = logging.getLogger("freelunch")

#User-Defined Imports:
import model

#Note: for some reason risknet.proc.[package_name] didn't work so I'm updating this yall :D
import sys
sys.path.append(r"src/risknet/proc") #reorient directory to access proc .py files
import label_prep
import reducer
import encoder

#Variables:
fm_root = "/Users/emily/Desktop/local_180/data/" #location of FM data files
data: List[Tuple[str, str, str]] = [('historical_data_time_2009Q1.txt', 'dev_labels.pkl', 'dev_reg_labels.pkl')]
cat_label: str = "default"
non_train_columns: List[str] = ['default', 'undefaulted_progress', 'flag']
#('historical_data_time_2014Q1.txt', 'oot_labels.pkl', 'oot_reg_labels.pkl')]

#Pipeline:

#Step 1: Label Processing: Returns dev_labels.pkl and dev_reg_labels.pkl
label_prep.label_proc(fm_root, data)

#Step 2: Reducer: Returns df of combined data to encode
df = reducer.reduce(fm_root, data[0]) 
#As of right now, we are only pulling 2009 data. So we only need data[0].

#However, if we want to add 2014 data in the future, we can add another Tuple(str,str,str) to the List data
#and uncomment this code:
#df = reducer.reduce(fm_root, data[1])

#Data Cleaning 1: Define datatypes; Define what should be null (e.g., which codes per column indicate missing data)

 #Define datatypes
df = encoder.datatype(df)

#Define where it should be null:
#where we have explicit mappings of nulls
df = encoder.num_null(df)

# where we have explicit mappings of nulls
df = encoder.cat_null(df)

#Data Cleaning 2: Categorical, Ordinal, and RME Encoding
# interaction effects
df['seller_servicer_match'] = np.where(df.seller_name == df.servicer_name, 1, 0)

'''Categorical Encoding'''
df = encoder.cat_enc(df)

'''Ordinal Encoding'''
df = encoder.ord_enc(df, fm_root)

'''RME Encoding'''
df = encoder.rme(df, fm_root)

#Data Cleaning 3: Remove badvars, scale
#Remove badvars (Feature filter). Save badvars into badvars.pkl, and goodvars (unscaled data) into
df = encoder.ff(df, fm_root) #Removes bad variables

#Scale the df
df = encoder.scale(df, fm_root)

#Training the XGB Model
data = model.xgb_train(fm_root, baseline=False)
auc, pr, recall = model.xgb_eval(data)

print(auc)
print(pr)
print(recall)