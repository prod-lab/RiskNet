import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pandas import DataFrame
import numpy as np
import sys
from typing import List, Dict, Tuple
import pickle
import logging
import yaml
import os

from risknet.proc import label_prep
from risknet.proc import reducer
from risknet.proc import encoder
from risknet.proc import parquet
from risknet.proc import fe
from risknet.run import model

class LoanDataset(Dataset):
   
    def __init__(self, features, labels):
        # Store the features and labels passed to the dataset
        self.features = features
        self.labels = labels
        
    def __len__(self):
        # Return label length
        return len(self.labels)
       
    def __getitem__(self, idx):
        # Convert to PyTorch tensors 
        return torch.tensor(self.features[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.float)
parquet.parquet_convert('historical_data_time_2009Q1.txt','historical_data_2009Q1.txt')
risknet_run_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'run')
sys.path.append(risknet_run_path)

risknet_proc_path = risknet_run_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'run')
sys.path.append(risknet_proc_path) #reorient directory to access proc .py files

config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'config','conf.yaml')
with open(config_path) as conf:
    config = yaml.full_load(conf)

#Variables:
fm_root = os.path.expanduser(config['data']['fm_root'])  #location of FM data files
data: List[Tuple[str, str, str]] = config['data']['files']
cat_label: str = "default"
non_train_columns: List[str] = ['default', 'undefaulted_progress', 'flag']

print(data)