#Global Imports:
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

#Default values: use feature engineered labels; do NOT focus on only credit score (baseline); use parquet
def pipeline(fe_enabled=True, baseline=False, p_true=True):
    '''
    This is where everything runs!
    We call various functions from .model, .reducer, .encoder, etc. step-by-step to run the pipeline.
    Check out the comments to see what each part of the code does.

    Parameters
    ----------
    fe_enabled: Boolean, optional
        Defines whether to run feature engineering on the dataframe or not.
        Default value is True

    baseline: Boolean, optional
        Defines whether to only use credit score (True) or use all features (False) to train/evaluate the model
        Default value is False
    
    p_true: Boolean, optional
        Defines whether to use data loaded from parquet (True) or use pandas data (False) to train/evaluate the model
        Default value is True

    Returns
    -------
    List[List[Float], List[Float], List[Float], Float] 
    Returns [auc, pr, recall, time] values as result from evaluating the model.

    Notes
    -------
    Loads data from the file path defined in config/conf.yaml. If this file path is not up-to-date, the pipeline will break.
    See other files for information on each function run.
    '''
    logger = logging.getLogger("freelunch")

    #This ensures the info-level logs get stored in a new file called "test.log"
    logging.basicConfig(
        filename="test.log",
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(message)s"
        )

    #load data
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


    #Note: for some reason risknet.proc.[package_name] didn't work so I'm updating this yall :D
    #sys.path.append(r"src/risknet/proc") #reorient directory to access proc .py files

    #Pipeline:

    #Pre-Step 1: Convert data to Parquet: 
    #Note: only need to do this once!
    #parquet.parquet_convert('historical_data_time_2009Q1.txt','historical_data_2009Q1.txt', fm_root)

    #Step 1: Label Processing: Returns dev_labels.pkl and dev_reg_labels.pkl (essential if p_true)
    #Note: if using parquet, we already labeled data with train/test/val flags in parquet.parquet_convert
    if not p_true:
        data = [('historical_data_time_2009Q1.txt','dev_labels.pkl', 'dev_reg_labels.pkl')]
        label_prep.label_proc(fm_root, data, p_true)

    #Step 2: Reducer: Returns df of combined data to encode
    #As of right now, we are only pulling 2009 data. So we only need data[0].
    df = reducer.reduce(fm_root, p_true)

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

    #Replace 'infinity' values with null values:
    df = encoder.inf_null(df)

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

    #Feature Engineering
    if fe_enabled:
        df = fe.fe(df, fm_root)

    #Training the XGB Model
    #print(df.columns)
    data, time = model.xgb_train(df, fm_root, baseline=baseline, cat_label='default')
    auc, pr, recall = model.xgb_eval(data)

    return [auc, pr, recall, time]