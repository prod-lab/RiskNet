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
import matplotlib.pyplot as plt

import sys

#User-Defined Imports:
#sys.path.append(r"src/risknet/run")
import model

#Note: for some reason risknet.proc.[package_name] didn't work so I'm updating this yall :D
sys.path.append(r"src/risknet/proc") #reorient directory to access proc .py files
import label_prep
import reducer
import encoder
import time 
#Variables:
fm_root = "../../../../../teams/a15-subgroup-2/data/" #location of FM data files
data: List[Tuple[str, str, str]] = [('historical_data_time_2009Q1.txt', 'dev_labels.pkl', 'dev_reg_labels.pkl')]
cat_label: str = "default"
non_train_columns: List[str] = ['default', 'undefaulted_progress', 'flag']
#('historical_data_time_2014Q1.txt', 'oot_labels.pkl', 'oot_reg_labels.pkl')]

#Pipeline:
start = time.time()
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

#Training the XGB Model
data = model.xgb_train(fm_root, baseline=True)
base_auc, base_pr, base_recall = model.xgb_eval(data)

print("XGBoost Model")
print(auc)
print(pr)
print(recall)
 
print("Baseline Model")
print(base_auc)
print(base_pr)
print(base_recall)

x = np.array(['XGtrain', 'Btrain', 'XGval', 'Bval', 'XGtest', 'Btest'])
ab = []
x_pos = [i for i, _ in enumerate(x)]

for i in range(3):
    ab.append(auc[i])
    ab.append(base_auc[i])


color = ['lightblue', 'lightblue', 'orange', 'orange', 'red', 'red']
edgecolor = ['green', 'purple', 'green', 'purple', 'green', 'purple']

'''AUC Curve'''
plt.bar(x, ab, alpha=0.8,\
        color = color,\
        edgecolor = edgecolor)

plt.ylabel("AUC")
plt.title("DSMLP Model AUC: \nXGBoost w/ Bayesian Optimization vs. Baseline (FICO Score)")

plt.xticks(x_pos, x)

plt.ylim(top=1)

plt.show()
fm_root2 = "/home/tambat/private/RiskNet/src/risknet/data/"
plt.savefig(fm_root2 + "xgb_auc.png")

plt.close()

'''Average Precision Score Chart: How well do we recognize positives?'''
ab_pr = []

for i in range(3):
    ab_pr.append(pr[i])
    ab_pr.append(base_pr[i])


plt.bar(x_pos, ab_pr, color=color, edgecolor=edgecolor, alpha=0.7)

plt.ylabel("Average Precision")
plt.title("DSMLP Model Average Precision: \nXGBoost w/ Bayesian Optimization vs. Baseline (FICO Score)")

plt.xticks(x_pos, x)

plt.ylim(top=.50)

plt.show()

plt.savefig(fm_root2 + "xgb_av_pr.png")

plt.close()

# '''Precision vs Recall Curve'''
# plt.plot(recall[1], pr[1])
# plt.plot(base_recall[1], base_pr[1])

# plt.legend(['XGB', 'Baseline'], loc='upper right')

# plt.title('PR Curve: XGBoost vs Baseline (FICO Score)')

# plt.xlabel('Recall')
# plt.ylabel('Precision')

# plt.show()
# plt.savefig(fm_root + "xgb_pr_curve.png")
# plt.close()

end = time.time()
print(end - start) 