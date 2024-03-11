'''This .py file defines functions and classes used to encode columns.'''

#Global Imports
import pandas as pd
import numpy as np
from typing import List, Dict
from pandas import DataFrame
import warnings
import pickle
#import dask.dataframe as dd #use dask in place of pandas


#User-Defined Imports
from risknet.proc import reducer


#Global Variables:
numericals: List[str] = ['credit_score', 'number_of_units', 'orig_combined_loan_to_value', 'dti_ratio', 'original_unpaid_principal_balance', 'original_ltv', 'number_of_borrowers']
categoricals: List[str] = ['first_time_homebuyer', 'occupancy_status', 'channel', 'prepayment_penalty_mortgage', 'product_type', 'property_type', 'loan_purpose', 'seller_name', 'servicer_name', 'super_conforming_flag']
non_train_columns: List[str] = ['default', 'undefaulted_progress', 'flag', 'loan_sequence_number']

#Functions:
#Define datatypes
def datatype(df):
    '''
    Sets numerical variables as type int64 and categorical variables as strings.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame passed in after reducer.reduce()

    Returns
    -------
    df : pandas.DataFrame
    '''
    df.loc[:, numericals] = df.loc[:, numericals].astype('int64')
    df.loc[:, categoricals] = df.loc[:, categoricals].astype(str)
    return df

def num_null(df):
    '''
    Defines null values for numerical columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame

    Returns
    -------
    pandas.DataFrame
        Modified DataFrame.
    '''
    numerical_null_map: Dict[str,int] = {'credit_score':9999, 'number_of_units':99, 'orig_combined_loan_to_value':999,
                            'dti_ratio':999, 'original_ltv':999, 'number_of_borrowers':99}
    for k,v in numerical_null_map.items():
        df[k] = np.where(df[k] == v, np.nan, df[k])
    return df

def cat_null(df):
    '''
    Defines null values for categorical columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame

    Returns
    -------
    pandas.DataFrame
        Modified DataFrame.
    '''
    categorical_null_map: Dict [str,str] = {'first_time_homebuyer':'9', 'occupancy_status': '9', 'channel':'9', 'property_type':'99', 'loan_purpose':'9'}
    for k,v in categorical_null_map.items():
        df[k] = np.where(df[k] == v, np.nan, df[k])
    return df

def inf_null(df):
    '''
    Replaces infinite values with null values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame

    Returns
    -------
    pandas.DataFrame
        Modified DataFrame.
    '''
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def cat_enc(df):
    '''
    Creates "was_missing" columns for each categorical column, giving binary 0/1 for missing/present values.
    Also replaces NA with "missing" in categorical columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        Modified DataFrame.
    '''
    for i in categoricals:
        df["was_missing_" + i] = np.where(df[i].isnull(), 1, 0)
    df[categoricals]: DataFrame = df[categoricals].fillna("missing")
    return df

def ord_enc(df, fm_root):
    '''
    Fits Ordinal Encoder on training data and ordinally encodes all columns.
    Also saves the ordinal encoder into a .pkl file for future use.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        Modified DataFrame.
    '''
    ordinal: OrdinalEncoder = OrdinalEncoder()
    ordinal.fit(df.loc[df.flag == 'train'], categoricals) #Fit encoder on train
    with open(fm_root + 'ordinal.pkl', 'wb') as f:
        pickle.dump(ordinal, f)

    df = pd.concat([df, ordinal.transform(df, categoricals)], axis=1)
    return df

def rme(df, fm_root, cat_label='default'):
    '''
    Performs regularized mean encoding on data in repository.
    Fits on training data and saves RME object in a pickle file.
    Applies encoding to all non-categorical data.
    For numerical data, creates "is_missing" columns documenting whether the value is missing,
    and fills NAs with 0.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    fm_root : str
        Location of data in repository.
    cat_label : str, optional
        Categorical columns in repository, defaults to 'default'.

    Returns
    -------
    pandas.DataFrame
        Modified DataFrame.
    '''
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

def ff(df, fm_root):
    '''
    Performs feature filtering (removing useless variables).
    Puts unused variables in "badvars.pkl" and remaining useful, unscaled variables in "df_unscaled.pkl".

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    fm_root : str
        Location of data.

    Returns
    -------
    pandas.DataFrame
        Modified DataFrame.
    '''
    train_columns: List[str] = [i for i in df.columns.to_list() if i not in non_train_columns]

    #Identify useless variables and save into badvars.pkl
    red = reducer.Reducer()
    badvars = list(red.feature_filter(df.loc[df['flag'] == 'train'].loc[:, train_columns])) #filter columns based on correlation using training data
    with open(fm_root + 'badvars.pkl', 'wb') as f:
        pickle.dump(badvars, f)
    #drop useless variables
    df = df.drop(columns=badvars)

    #save useful variables (remaining, unscaled) into df_unscaled.pkl
    with open(fm_root + 'df_unscaled.pkl', 'wb') as f:
        pickle.dump(df, f)
    return df

def scale(df, fm_root):
    '''
    Scales the dataset and saves min/max/scaled DataFrame into .pkl files.
    Stores the final scaled DataFrame in df.pkl.
    Returns the scaled DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    fm_root : str
        Location of data.

    Returns
    -------
    pandas.DataFrame
        Modified DataFrame.
    '''
    train_columns: List[str] = [i for i in df.columns.to_list() if i not in non_train_columns]
    #train_columns.remove('loan_sequence_number') #This is not a numerical column so it can't be scaled with min/max subtraction

    train_mins = inf_null(df.loc[df['flag'] == 'train'].loc[:, train_columns]).min()

    with open(fm_root + 'train_mins.pkl', 'wb') as f:
        pickle.dump(train_mins, f)

    train_maxs = inf_null(df.loc[df['flag'] == 'train'].loc[:, train_columns]).max()

    with open(fm_root + 'train_maxs.pkl', 'wb') as f:
        pickle.dump(train_maxs, f)

    #EC Note: wrapped inf_null around this to make sure no infinite values in columns
    df.loc[:, train_columns] = inf_null((df.loc[:, train_columns] - train_mins) / (train_maxs - train_mins)) #Scaling values
    df.dropna(axis=1, how='all', inplace=True) #Drop any columns that are ONLY comprised of NaN values

    #Store dataframes and labels
    with open(fm_root + 'df.pkl', 'wb') as f:
        pickle.dump(df, f)
    
    return df

#Classes:
class RobustHot:
    '''
    Encode categorical integer features as a one-hot numeric array.

    This class transforms categorical features to a one-hot numeric array, where each
    categorical feature is transformed into a new binary column (one-hot vector) indicating
    the presence of that feature in the original sample.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data to encode
    cols_to_transform: List[str]
        List containing the names of columns to one-hot encode
    sep: str = "__"
        String containing the separator when creating new column names for encoded data
    dummy_na: bool = True
        Boolean defining whether to create a dummy variable for NA values
    drop_first: bool = False
        Boolean defining whether to drop the first category level in each feature
    return_all: bool = False
        If True, return both encoded and non-encoded categorical values. If false, only return the robust-encoded columns.

    Attributes
    ----------
    self.cat_dummies : list of str, default=[]
        List containing the names of the categorical columns to iterate over
    self.processed_columns : str, default=''
        Name of the processed columns in the output DataFrame.
    self.sep : str, default='__'
        Separator used to concatenate feature names with category names in the generated dummy variables.
    self.dummy_na : bool, default=True
        Whether to create a dummy variable for NA values. If True, a column with all zeros is created
        for NA values in the original categorical variable.
    self.drop_first : bool, default=True
        Whether to drop the first category level in each feature to avoid multicollinearity. If True,
        the first category level is dropped and only n-1 levels are encoded.
    self.cols_to_transform : list of str, default=[]
        List containing the names of columns to be transformed using one-hot encoding.

    Methods
    -------
    fit_transform(self, df: DataFrame, cols_to_transform: List[str], sep: str = "__", dummy_na: bool = True,
                      drop_first: bool = False, return_all: bool = False)
        Fit the OneHotEncoder to the given data.
    transform(self, df, return_all=False)
        Transform the given data using one-hot encoding and return encoded DataFrame.
    '''
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
    '''
    Scale the continuous integer features using RME.

    This class scales continuous numeric features based on RME, where each
    numeric feature is scaled to show frequency.

    Parameters
    ----------
    df: pd.DataFrame
        Contains the numerical data to be scaled
    targetLabel: str
        Contains the name of the target/y variable. Usually set as "default" 
    colsToTransform: List[str]
        Contains the names of numerical columns to scale
    a: Int
        Alpha value to adjust RME values. Default value 1.
    earlyStop: Int.
        Defines how many rounds should traverse before early stopping. Default value is None
    defaultPrior: Int.
        Defines the mean/scaling value if not None. Default value None

    Attributes
    ----------
    self.levelDict: Dict = {}
        Each key saves the dataframe's column. The value corresponds to that column's mean value.
    self.nan: float = np.nan
        Defines the NA value for every column. Default value is np.nan
    self.defaultPrior: float = None
        Defines the mean to regularize on. Default value is None and will be defined in fit()

    Methods
    -------
    fit(self, df, targetLabel, colsToTransform, a=1, earlyStop=None, defaultPrior=None)
        Fits the RME using either the defaultPrior or the calculated mean of each column. Not currently used in pipeline.
    fit_faster(self, df, targetLabel, colsToTransform, a=1, early_stop=None, default_prior=None)
        Fits the RME "faster". Does not take early stopping as a possibility. Currently used in pipeline.
    transform(self, transformFrame, colsToTransform)
        Uses fitted RME object to transform the dataframe columns. Returns modified dataframe
    '''
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
    '''
    Encode ordinal categorical features as an ordinal value.

    This class transforms categorical features to an ordinal value, where each
    categorical value is transformed into an integer from 1 - (n-1) indicating
    the ordinal value. (n = number of ordinal levels in the variable)

    Parameters
    ----------
    df: DataFrame
        Contains ordinal columns to fit and transform upon
    cols_to_fit: List[str]
        Defines the names of columns to fit the encoding object on
    rare_high = True
        Defines whether to perform ordinal encoding and add a column for None values. If not, all values are 0. Default value is None.
    missing_name = "XXXXXX"
        Defines the [str] that the encoder use as a column name indicate that it counts null values. Default value is "XXXXXX"
    cols_to_transform: List[str]
        Defines the names of columns to transform using encoding object

    Attributes
    ----------
    self.level_dict = {}
        Stores the index (sorted by frequency) of each ordinal value.
    self.rare_high: Boolean
        Defines whether to perform ordinal encoding and add a column for None values. If not, all values are 0. Default value is None.
    self.missing_name = str
        Defines the [str] that the encoder use as a column name indicate that it counts null values. Default value is None
    self.element_length = None
        Defines how many unique ordinal objects are in a given column. Default value is None.

    Methods
    -------
    fit(self, df: DataFrame, cols_to_fit: List[str], rare_high = True, missing_name = "XXXXXX"):fit(self, df: DataFrame, cols_to_fit: List[str], rare_high = True, missing_name = "XXXXXX")
        Fits the ordinal encoder using the dataframe and cols_to_fit.
    transform(self, df: DataFrame, cols_to_transform: List[str])
        Transforms the cols_to_transform in df.
    '''
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
    

