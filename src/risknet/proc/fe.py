'''Uses featuretools (API) to engineer new features'''

import featuretools as ft
import pickle
import pandas as pd

import logging
logger = logging.getLogger("freelunch")

#Local imports
#from risknet.proc import encoder

'''
Uses featuretools's DFS method to generate new features from df.pkl

input: df = DataFrame pulled from df.pkl that contains all cleaned/formatted features
output: returns DataFrame with multiple generated features
'''
def fe(df, fm_root):
    #Create an EntitySet that stores dataframe
    es = ft.EntitySet(id="data")
    es = es.add_dataframe(
        dataframe_name="data",
        dataframe=df,
        index="loan_sequence_number",
        time_index="first_payment_date",
    )
    
    #This returns feature_matrix (new features generated off of df) and defines features
    feature_matrix, feature_defs = ft.dfs(entityset=es, 
                                          target_dataframe_name="data",
                                          ignore_columns={"data": ["flag", "default"]},
                                          max_depth=3)
                                          #Ignoring flag (for train/test/valid), "default" (for y classification)

    #Log feature definitions
    logger.info("Feature definitions:")
    logger.info(feature_defs)

    #Combine new features + old feature table (df)
    #combo = feature_matrix.merge(
    #        df, on="loan_sequence_number", how="inner")
    combo = feature_matrix
    
    #Do not want to create new features based on pred
    #TODO: can we keep undefaulted_progress x and y? Experiment
    #combo = combo.drop(['flag_x', 'flag_y', 'default_x', 'default_y', 'undefaulted_progress_x', 'undefaulted_progress_y'], axis=1) 
    combo = combo.dropna(axis='columns')

    with open(fm_root + 'combo.pkl', 'wb') as f:
            pickle.dump(combo, f)

    return combo