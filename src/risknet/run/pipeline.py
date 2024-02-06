
from memory_profiler import profile

@profile
def main():
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
    import yaml
    import os 
    logger = logging.getLogger("freelunch")

    #This ensures the info-level logs get stored in a new file called "test.log"
    logging.basicConfig(
        filename="test.log",
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s"
        )

    import sys

    #User-Defined Imports:
    risknet_run_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'run')
    sys.path.append(risknet_run_path)

    from risknet.run import model
    #sys.path.append(r"src/risknet/run")

    #Note: for some reason risknet.proc.[package_name] didn't work so I'm updating this yall :D
    risknet_proc_path = risknet_run_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'run')
    sys.path.append(risknet_proc_path) #reorient directory to access proc .py files
    from risknet.proc import label_prep
    from risknet.proc import reducer
    from risknet.proc import encoder
    from risknet.proc import fe
    from risknet.sys import cpu_monitor

    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'config','conf_template.yaml')
    with open(config_path) as conf:
        config = yaml.full_load(conf)

    #Get information on system baselines:
    logger.info("System baseline: ")
    logger.info(cpu_monitor.baseline())

    #Variables:
    fm_root = os.path.expanduser(config['data']['fm_root'])  #location of FM data files
    data: List[Tuple[str, str, str]] = config['data']['files']
    cat_label: str = "default"
    non_train_columns: List[str] = ['default', 'undefaulted_progress', 'flag']
    #('historical_data_time_2014Q1.txt', 'oot_labels.pkl', 'oot_reg_labels.pkl')]

    #Start: set CPU to zero
    cpu_monitor.cpu_check()

    #Pipeline:
    #Step 1: Label Processing: Returns dev_labels.pkl and dev_reg_labels.pkl
    #label_prep.label_proc(fm_root, data)
    label_prep.execute(fm_root)

    #Step 2: Reducer: Returns df of combined data to encode
    df = reducer.reduce(fm_root, data[0]) 
    #As of right now, we are only pulling 2009 data. So we only need data[0].

    #However, if we want to add 2014 data in the future, we can add another Tuple(str,str,str) to the List data
    #and uncomment this code:
    #df = reducer.reduce(fm_root, data[1])
    logger.info("CPU usage after pulling/combining data: " + str(cpu_monitor.cpu_check()))

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
    #df = encoder.ff(df, fm_root) #Removes bad variables

    #Scale the df
    df = encoder.scale(df, fm_root)
    #This puts the complete scaled/cleaned dataframe into df.pkl located in fm_root

    logger.info("CPU usage after data cleaning: " + str(cpu_monitor.cpu_check()))

    #Feature Engineering
    #df = fe.fe(df, fm_root)
    df = fe.fe(df, fm_root)
    #fe = pd.read_pickle(fm_root + 'combo.pkl')
    #print(fe.info(verbose=True))

    logger.info("CPU usage after feature engineering: " + str(cpu_monitor.cpu_check()))

    #Training the XGB Model
    data = model.xgb_train(df, fm_root, baseline=False)
    auc, pr, recall = model.xgb_eval(data)

    logger.info("Train, val, test recall: " + str(auc))
    logger.info("Train, val, test precision: " + str(pr))
    logger.info("Train, val, test recall: " + str(recall))
