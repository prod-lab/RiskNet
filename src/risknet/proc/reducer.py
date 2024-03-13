'''
This .py file currently concatenates the Freddie Mac file with the labels we created in label_prep.py.
It also defines the Reducer class where we use ts split (timeseries split) t 
'''

from pandas import DataFrame
import pandas as pd
import dask.dataframe as dd
import numpy as np
from typing import List, Tuple
import pyarrow.parquet as pq
from datetime import timedelta, date
from sklearn.model_selection import train_test_split

def reduce(fm_root, p_true, test_size=1_000, split_ratio=[0.9, 0.1]):
    '''
    For a given year and that year's labels and progress, concatenates all three datasets together
    and returns a complete dataset.

    Parameters
    ----------
    fm_root : str
        A location where data is held.
    p_true : bool, optional
        Determines whether to use the entire parquet-loaded dataset (True) or the first 10 million rows of data using pandas (False).
    test_size : int, optional
        The size (number of rows) for the test set.
    split_ratio : list of float, optional
        The train/validation ratio of the dataset.

    Returns
    -------
    pandas.DataFrame
        Modified DataFrame.

    Notes
    -------
    - Creates a test subset of the [test_size] most recent loan instances. Then uses parquet or pandas to load data and split into train/val.
    - Assumes existence of org.parquet (generated from parquet.py), dev_labels.parquet, and dev_reg_labels.parquet (generated from label_prep.py) in the fm_root location.
    - Assumes .parquet and .txt files have the same columns (which they should!)
    '''
    origination_cols: List[str] = ["credit_score", "first_payment_date", "first_time_homebuyer", "maturity_date",
                                "metropolitan_division", "mortgage_insurance_percent", "number_of_units",
                                "occupancy_status", "orig_combined_loan_to_value", "dti_ratio",
                                "original_unpaid_principal_balance", "original_ltv", "original_interest_rate",
                                "channel", "prepayment_penalty_mortgage", "product_type", "property_state",
                                "property_type", "postal_code", "loan_sequence_number", "loan_purpose",
                                "original_loan_term",
                                "number_of_borrowers", "seller_name", "servicer_name", "super_conforming_flag", "row_hash"]

    drop_cols: List[str] = ['maturity_date', 'metropolitan_division', 'original_interest_rate', 'property_state',
                            'postal_code', 'mortgage_insurance_percent', 'original_loan_term']
    
    #Read the parquet file to get the entire dataset
    temp = pd.read_parquet(fm_root + 'org.parquet')
    temp.columns = origination_cols
    temp = temp.drop(columns = "row_hash")

    #Create a separate test subset from entire dataset by finding the most recent entries (sorted by first_payment_date)
    #The size of this subset will be test_size
    test_index = temp.shape[0] - test_size
    temp = temp.sort_values(by=['first_payment_date']) #Indices should stay the same even after we sort!
    test_cond = [temp.loc[:, 'first_payment_date'].rank(method='first') <= test_index, temp.loc[:, 'first_payment_date'].rank(method='first') > test_index]
    choices = [np.NaN, 'test']
    temp['flag'] = np.select(test_cond, choices, default=np.nan)
    #This will create a column ['flag'] which either indicates 'test' or NaN.

    #Create the separate test dataset. Drop 'drop_cols
    test = temp.loc[temp['flag'] == 'test'].drop(columns=drop_cols)
    #Merge with the appropriate files so it gets `default` label
    test = test.merge(
                pd.read_pickle(fm_root + 'dev_labels.parquet'), on="loan_sequence_number",
                how="inner").merge(
                pd.read_pickle(fm_root + 'dev_reg_labels.parquet'), on="loan_sequence_number",
                how="inner")
    test_indices = list(test.index) #Use test_indices to make sure pandas doesn't read these lines into train/val

    #Now remove the test subset from the parquet dataset so we can split train/val:
    trainval = temp.loc[temp['flag'] != 'test']
    
    #Then we pull train and val data from historical_data_2009Q1!
    if p_true:
        df = pd.concat([Reducer.em_simple_ts_split(trainval.merge(
                pd.read_parquet(fm_root + 'dev_labels.parquet'), on="loan_sequence_number",
                how="inner").merge(
                pd.read_parquet(fm_root + 'dev_reg_labels.parquet'), on="loan_sequence_number",
                how="inner").drop(columns=drop_cols).drop(columns=['flag']), sort_key='first_payment_date', 
                split_ratio=split_ratio)])
    else:
        df = pd.concat([Reducer.em_simple_ts_split(pd.read_csv(fm_root + "historical_data_2009Q1.txt", sep='|', index_col=False, skiprows=test_indices, nrows=10_000_000, names=origination_cols).merge(
            pd.read_pickle(fm_root + 'dev_labels.pkl'), on="loan_sequence_number", how="inner").merge(
            pd.read_pickle(fm_root + 'dev_reg_labels.pkl'), on="loan_sequence_number", how="inner").drop(columns=drop_cols), 
            sort_key='first_payment_date', split_ratio=split_ratio)])
    
    #Re-add the test set at the bottom of the dataframe
    df = pd.concat([df, test], join="inner")
    return df

class Reducer:
    '''
    Filters out highly-correlated variables (feature_filter()) and splits the data into train/val.

    Parameters
    ----------
    df: DataFrame
        Dataset (not including test data, which has been removed in reduce())

    sort_key: str
        String that defines the name of the sorting variable (time variable). Should be `first_payment_date`.
    
    split_ratio: List[Float], optional.
        Defines the split of train/val.
        Assumes that the list has length 2 and sums up to 1.0. Uses the first value to split the dataset.
        Default value is [0.9, 0.1]

    Attributes
    ----------
    self.varsToRemove = List[str]
        List of unique column values that either have low standard deviation, high correlation, or high amounts of nulls.
        Default value is [].

    Methods
    -------
    feature_filter(self, df, max_null_ratio=0.7, zero_var_threshold=0.0000000000001, run_correlations=True,
                       corr_threshold=0.70)
    Filters out variables that are highly-correlated, have too many null values, or have low standard deviation.
    Returns a list of variables to remove.

    filter_split(txn_root: str, timestamp:str, drop: List[str], filter_exprs: Tuple[Tuple[str,str]] = (("", ""),),
                     aged_interval: int = 70, oot_requirement: int = 120, max_lookback_days: int = 180)
    Splits the dataset into train/val/test given a max lookback window. Deprecated/currently not in use in pipeline.

    simple_ts_split(df: DataFrame, sort_key: str, split_ratio: list = [0.8, 0.1, 0.1])
    Splits the entire dataset into train/val/test. Deprecated/currently not in use.

    em_simple_ts_split(df: DataFrame, sort_key: str, split_ratio: list = [0.9, 0.1])
    Splits the entire dataset into train/val. Assumes that test has already been separated out. Currently in use.
    
    def random_split(df: DataFrame, split_ratio: float = .8):
    Splits the entire dataset randomly. Deprecated/currently not in use.
    '''
    def __init__(self):
        self.varsToRemove = []

    def feature_filter(self, df, max_null_ratio=0.7, zero_var_threshold=0.0000000000001, run_correlations=True,
                       corr_threshold=0.70):

        null_ratios = df.isnull().sum() / len(df)
        high_nulls = null_ratios[null_ratios >= max_null_ratio].index.values.tolist()

        zero_var_index = df.std() < zero_var_threshold
        zero_vars = zero_var_index[zero_var_index == True].index.values.tolist()

        high_corr = []

        # TODO make this return only lower variance high corr variables

        if run_correlations:
            corr_matrix = df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool)) #Changed from numpy.bool to bool due to deprecation error
            high_corr = [column for column in upper.columns if any(upper[column] > corr_threshold)]

        self.varsToRemove = list(set(high_nulls + zero_vars + high_corr))
        return self.varsToRemove
    
    '''
    @staticmethod
    def filter_split(txn_root: str, timestamp:str, drop: List[str], filter_exprs: Tuple[Tuple[str,str]] = (("", ""),),
                     aged_interval: int = 70, oot_requirement: int = 120, max_lookback_days: int = 180) -> DataFrame:

        txn: DataFrame = pq.ParquetDataset(
            txn_root).read_pandas().to_pandas().drop(columns=drop)

        txn = pd.concat([x_train, y_train, x_test, y_test, x_val, y_val])
        if filter_exprs:
            for i in filter_exprs:
                txn: DataFrame = txn[eval(i[0])].drop(columns=eval(i[1]), errors='ignore')

        first_full_agg_date: str = (txn.loc[:, timestamp].min() + np.timedelta64(max_lookback_days, 'D')).strftime('%Y-%m-%d')

        last_aged_date: str = (date.today() - timedelta(days=aged_interval)).strftime('%Y-%m-%d')

        validation_begin_date: str = (
            (date.today() - timedelta(days=aged_interval)) - timedelta(days=oot_requirement)).strftime('%Y-%m-%d')

        training_end_date: str = (
            ((date.today() - timedelta(days=aged_interval)) - timedelta(days=oot_requirement)) - timedelta(days=aged_interval)).strftime('%Y-%m-%d')

        txn_dev: DataFrame = txn[(txn[timestamp] >= first_full_agg_date) & (txn[timestamp] < training_end_date)]

        txn_dev_train: DataFrame = txn_dev.sort_values(by=timestamp).iloc[0: round(txn_dev.shape[0] * .8), :]

        txn_dev_train.loc[:, 'flag'] = 'train'

        txn_dev_test: DataFrame = txn_dev.sort_values(by=timestamp).iloc[round(txn_dev.shape[0] * .8):, :]

        txn_dev_test.loc[:, 'flag'] = 'test'

        txn_val: DataFrame = txn[(txn[timestamp] >= validation_begin_date) & (txn[timestamp] < last_aged_date)]

        txn_val.loc[:, 'flag'] = 'val'

        txn_return: DataFrame = pd.concat([txn_dev_train, txn_dev_test, txn_val])

        return txn_return
    '''

    @staticmethod #MODIFIED BY EC TO GET TRAIN/TEST/VALIDATION
    def simple_ts_split(df: DataFrame, sort_key: str, split_ratio: list = [0.8, 0.1, 0.1]):
      first_div = split_ratio[0]
      second_div = split_ratio[0] + split_ratio[1]
      df = df.sort_values(by=[sort_key])

      conditions = [df.loc[:, sort_key].rank(pct=True, method='first') <= first_div, \
       (df.loc[:, sort_key].rank(pct=True, method='first') > first_div) & (df.loc[:, sort_key].rank(pct=True, method='first') < second_div), \
                    df.loc[:, sort_key].rank(pct=True, method='first') > second_div]
      choices     = [ "train", 'test', 'val' ]

      df['flag'] = np.select(conditions, choices, default=np.nan)
      #df['flag'] = np.where(df.loc[:, sort_key].rank(pct=True, method='first') <= split_ratio, 'train', 'test')
      return df

    @staticmethod #MODIFIED BY EC TO GET TRAIN/TEST/VALIDATION and consistently-sized test size
    def em_simple_ts_split(df: DataFrame, sort_key: str, split_ratio: list = [0.9, 0.1]):
      first_div = split_ratio[0]
      df = df.sort_values(by=[sort_key])

      conditions = [df.loc[:, sort_key].rank(pct=True, method='first') <= first_div, (df.loc[:, sort_key].rank(pct=True, method='first') > first_div)]
      choices     = ['train', 'val']
      df['flag'] = np.select(conditions, choices, default=np.nan)
      print("train size: ", str(df['flag'].value_counts()['train']), ", val size: ", str(df['flag'].value_counts()['val']))
      return df

    @staticmethod
    def random_split(df: DataFrame, split_ratio: float = .8):
        train, test = train_test_split(df, test_size=1 - split_ratio)
        train['flag'] = 'train'
        test['flag'] = 'test'
        df = pd.concat([train, test])
        return df
