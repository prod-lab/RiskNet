#Global Imports:
import pandas as pd
from pandas import DataFrame

import numpy as np

from typing import List, Dict, Tuple
import pickle
import logging
logger = logging.getLogger("freelunch")

#User-Defined Imports:
import label_prep
import data_prep
import reducer
import encoder
import model


fm_root = "/Users/zhen/Desktop/dsc180a-data/" 
data = [('historical_data_time_2009Q1.txt', 'dev_labels.pkl', 'dev_reg_labels.pkl')]
cat_label: str = "default"
non_train_columns: List[str] = ['default', 'undefaulted_progress', 'flag']

data_prep.data_processing(fm_root, data)

