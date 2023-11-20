'''This .py file defines functions and classes used to encode columns.'''

#Global Imports
import pandas as pd
import numpy as np
from typing import List, Dict
from pandas import DataFrame
import warnings
import pickle
import dask.dataframe as dd #use dask in place of pandas


#User-Defined Imports
import reducer

#Global Variables:
numericals: List[str] = ['credit_score', 'number_of_units', 'orig_combined_loan_to_value', 'dti_ratio', 'original_unpaid_principal_balance', 'original_ltv', 'number_of_borrowers']
categoricals: List[str] = ['first_time_homebuyer', 'occupancy_status', 'channel', 'prepayment_penalty_mortgage', 'product_type', 'property_type', 'loan_purpose', 'seller_name', 'servicer_name', 'super_conforming_flag']
non_train_columns: List[str] = ['default', 'undefaulted_progress', 'flag', 'loan_sequence_number']

#Functions:
#Define datatypes
'''
datatype: sets numerical variables as type int64 and categorical variables as strings
input:
- df (DataFrame): passed in after Reducer
'''
def datatype(df):
    df.loc[:, numericals] = df.loc[:, numericals].astype('int64')
    df.loc[:, categoricals] = df.loc[:, categoricals].astype(str)
    return df

'''
num_null: defines null values for numerical columns
input:
- df (DataFrame): passed in after Reducer
'''
def num_null(df):
    numerical_null_map: Dict[str,int] = {'credit_score':9999, 'number_of_units':99, 'orig_combined_loan_to_value':999,
                            'dti_ratio':999, 'original_ltv':999, 'number_of_borrowers':99}
    for k,v in numerical_null_map.items():
        df[k] = np.where(df[k] == v, np.nan, df[k])
    return df

'''
cat_null: defines null values for categorical columns
input:
- df (DataFrame): passed in after Reducer
'''
def cat_null(df):
    categorical_null_map: Dict [str,str] = {'first_time_homebuyer':'9', 'occupancy_status': '9', 'channel':'9', 'property_type':'99', 'loan_purpose':'9'}
    for k,v in categorical_null_map.items():
        df[k] = np.where(df[k] == v, np.nan, df[k])
    return df

'''
cat_enc: creates "was_missing" columns for each categorical giving binary 0/1 missing/present; also replaces NA with missing in categorical cols
input: 
- df (DataFrame)
- cat_label (str): usually 'default'.
'''
def cat_enc(df, cat_label='default'):
    for i in categoricals:
        df["was_missing_" + i] = np.where(df[i].isnull(), 1, 0)
    df[categoricals]: DataFrame = df[categoricals].fillna("missing")
    return df

'''
ord_enc: fits Ordinal Encoder on training data and ordinally encodes all columns. Also puts ordinal encoder into a .pkl file for future use.
input: df (DataFrame)
'''
def ord_enc(df, fm_root):
    ordinal: OrdinalEncoder = OrdinalEncoder()
    ordinal.fit(df.loc[df.flag == 'train'], categoricals) #Fit encoder on train
    with open(fm_root + 'ordinal.pkl', 'wb') as f:
        pickle.dump(ordinal, f)

    df = pd.concat([df, ordinal.transform(df, categoricals)], axis=1)
    return df

'''
rme:
inputs:
- df (DataFrame)
- fm_root (str): location of data in repository
- cat_label (str): categorical columns in repository. Usually 'default'.
'''
def rme(df, fm_root, cat_label='default'):
    '''Regularized Mean Encoding'''
    rme = RegularizedMeanEncoder()
    rme.fit_faster(df.loc[df['flag'] == 'train'].loc[:,
                        [cat_label] + categoricals],
                        targetLabel=cat_label, colsToTransform=categoricals)
    
    #save RME into .pkl
    with open(fm_root + 'rme.pkl', 'wb') as f:
        pickle.dump(rme, f)

    #apply RME on df
    df: DataFrame = pd.concat([df, rme.transform(df, categoricals)], axis=1).drop(columns=categoricals)

    for i in numericals:
        df["was_missing_" + i] = np.where(df[i].isnull(), 1, 0)

    df[numericals]: DataFrame = df[numericals].fillna(0)

    return df
'''
ff: feature filtering (removing useless variables)
inputs:
- df (dataframe)
- fm_root (location of data)
'''
def ff(df, fm_root):
    train_columns: List[str] = [i for i in df.columns.to_list() if i not in non_train_columns]

    #Identify useless variables and save into badvars.pkl
    red = reducer.Reducer()
    badvars = list(red.feature_filter(df.loc[df['flag'] == 'train'].loc[:, train_columns]))
    with open(fm_root + 'badvars.pkl', 'wb') as f:
        pickle.dump(badvars, f)
    #drop useless variables
    df = df.drop(columns=badvars)

    #save useful variables (remaining, unscaled) into df_unscaled.pkl
    with open(fm_root + 'df_unscaled.pkl', 'wb') as f:
        pickle.dump(df, f)
    return df
'''
scale: scaling the dataset and saving min/max/scaled dataframe into .pkls
inputs:
- df (dataframe)
- fm_root (location of data)
'''
def scale(df, fm_root):
    '''Scaling'''
    train_columns: List[str] = [i for i in df.columns.to_list() if i not in non_train_columns]
    #train_columns.remove('loan_sequence_number') #This is not a numerical column so it can't be scaled with min/max subtraction

    train_mins = df.loc[df['flag'] == 'train'].loc[:, train_columns].min()

    with open(fm_root + 'train_mins.pkl', 'wb') as f:
        pickle.dump(train_mins, f)

    train_maxs = df.loc[df['flag'] == 'train'].loc[:, train_columns].max()

    with open(fm_root + 'train_maxs.pkl', 'wb') as f:
        pickle.dump(train_maxs, f)

    df.loc[:, train_columns] = (df.loc[:, train_columns] - train_mins) / (train_maxs - train_mins) #Scaling values

    '''Store dataframes and labels'''

    with open(fm_root + 'df.pkl', 'wb') as f:
        pickle.dump(df, f)
    
    return df

#Classes:
class RobustHot:
    def __init__(self):

        self.cat_dummies: List[str] = []
        self.processed_columns: str = []
        self.sep: str = "__"
        self.dummy_na: bool = True
        self.drop_first: bool = True
        self.cols_to_transform: List[str] = []

    def fit_transform(self, df: DataFrame, cols_to_transform: List[str], sep: str = "__", dummy_na: bool = True,
                      drop_first: bool = False, return_all: bool = False):

        df_processed: DataFrame = pd.get_dummies(df, prefix_sep=sep, dummy_na=dummy_na, drop_first=drop_first,
                                                 columns=cols_to_transform)

        self.sep = sep
        self.dummy_na = dummy_na
        self.drop_first = drop_first
        self.cols_to_transform = cols_to_transform

        self.cat_dummies = [col for col in df_processed if sep in col and col.split(sep)[0] in cols_to_transform]

        self.processed_columns = list(df_processed.columns[:])

        if return_all:
            return df_processed.loc[:, self.processed_columns]

        else:
            return df_processed.loc[:, self.cat_dummies]

    def transform(self, df, return_all=False):

        df_test_processed: DataFrame = pd.get_dummies(df, prefix_sep=self.sep, dummy_na=self.dummy_na,
                                                      drop_first=self.drop_first,
                                                      columns=self.cols_to_transform)

        for col in df_test_processed.columns:
            if (self.sep in col) and (col.split(self.sep)[0] in self.cols_to_transform) and col not in self.cat_dummies:
                print("Removing unseen feature {}".format(col))
                df_test_processed.drop(col, axis=1, inplace=True)

        for col in [t for t in self.processed_columns if self.sep in t]:
            if col not in df_test_processed.columns:
                print("Adding missing feature {}".format(col))
                df_test_processed[col] = 0

        if return_all:
            return df_test_processed.loc[:, self.processed_columns]
        else:
            return df_test_processed.loc[:, self.cat_dummies]


class RegularizedMeanEncoder:

    def __init__(self):

        self.levelDict: Dict = {}
        self.nan: float = np.nan
        self.defaultPrior: float = None

    def fit(self, df, targetLabel, colsToTransform, a=1, earlyStop=None, defaultPrior=None):

        if defaultPrior == None:
            self.defaultPrior = df[targetLabel].mean()
        else:
            self.defaultPrior = defaultPrior

        for i in colsToTransform:
            self.levelDict[i] = {}
            for l in df[i].unique():
                if l == self.nan:
                    warnings.warn(
                        "There are missing values in " + str(i) + ".  Consider converting this to its own level.")
                self.levelDict[i][l] = self.defaultPrior

        for column in colsToTransform:
            for category in self.levelDict[column].keys():
                for i, level in enumerate(df.loc[df[column] == category, :][column]):
                    if i == 0:
                        pass
                    elif i == earlyStop:
                        break
                    else:
                        self.levelDict[column][category] = (df.loc[df[column] == category, :].iloc[0:i][
                                                                targetLabel].sum() + (a * self.defaultPrior)) / (
                                                                       df.loc[df[column] == category, :].iloc[
                                                                       0:i].shape[0] + a)

    def fit_faster(self, df, targetLabel, colsToTransform, a=1, early_stop=None, default_prior=None):

        if default_prior is None:
            self.defaultPrior = df[targetLabel].mean()
        else:
            self.defaultPrior = default_prior

        for i in colsToTransform:
            self.levelDict[i] = {}
            for level in df[i].unique().tolist():
                if level == self.nan:
                    warnings.warn(
                        "There are missing values in " + str(i) + ".  Consider converting this to its own level.")
                self.levelDict[i][level] = self.defaultPrior

        for column in colsToTransform:
            for category in self.levelDict[column].keys():
                halt = df.loc[df[column] == category, :][column].shape[0]
                self.levelDict[column][category] = (df.loc[df[column] == category, :].iloc[0:(halt - 1)][
                                                        targetLabel].sum() + (a * self.defaultPrior)) / (
                                                           df.loc[df[column] == category, :].iloc[0:(halt - 1)].shape[
                                                               0] + a
                                                   )

    def transform(self, transformFrame, colsToTransform):
        returnFrame = pd.DataFrame(index=transformFrame.index)

        for i in colsToTransform:
            returnFrame[i + "_enc"] = transformFrame[i].map(self.levelDict[i]).fillna(self.defaultPrior)

        return returnFrame


class OrdinalEncoder:
    def __init__(self):
        self.level_dict = {}
        self.rare_high = None
        self.missing_name = None
        self.element_length = None

    def fit(self, df: DataFrame, cols_to_fit: List[str], rare_high = True, missing_name = "XXXXXX"):
        self.rare_high = rare_high
        self.missing_name = missing_name
        for i in cols_to_fit:
            if rare_high:
                element_list = df[i].value_counts().sort_values(ascending=False).index.to_list() + [self.missing_name]
                self.level_dict[i] = {k: element_list.index(k) for k in element_list}
            else:
                element_list = df['a'].value_counts().sort_values(ascending=True).index.to_list()
                self.element_length = len(element_list)
                self.level_dict[i] = {k: element_list.index(k) for k in element_list}

    def transform(self, df: DataFrame, cols_to_transform: List[str]):
        return_frame = pd.DataFrame(index=df.index)

        if self.rare_high:
            for i in cols_to_transform:
                return_frame[i + "_ord_enc"] = df[i].map(self.level_dict[i]).fillna(self.level_dict[i][self.missing_name])
        else:
            for i in cols_to_transform:
                return_frame[i + "_ord_enc"] = df[i].map(self.level_dict[i]).fillna(self.element_length)
        return return_frame
    

