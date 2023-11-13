'''
This .py file currently concatenates the Freddie Mac file with the labels we created in label_prep.py.
It also defines the Reducer class where we use ts split (timeseries split) t 
'''

from pandas import DataFrame
import pandas as pd
import numpy as np
from typing import List, Tuple
import pyarrow.parquet as pq
from datetime import timedelta, date
from sklearn.model_selection import train_test_split

'''
reduce: for a given year + that year's labels and progress, concats all 3 datasets together and returns a complete dataset.
input:
- fm_root: str: a location where data is held
- i: Tuple(str, str, str): holds the file names for [(fm_dataset, default_label.pkl, default_progress.pkl)]
'''
def reduce(fm_root, i):
    origination_cols: List[str] = ["credit_score", "first_payment_date", "first_time_homebuyer", "maturity_date",
                                "metropolitan_division", "mortgage_insurance_percent", "number_of_units",
                                "occupancy_status", "orig_combined_loan_to_value", "dti_ratio",
                                "original_unpaid_principal_balance", "original_ltv", "original_interest_rate",
                                "channel", "prepayment_penalty_mortgage", "product_type", "property_state",
                                "property_type", "postal_code", "loan_sequence_number", "loan_purpose",
                                "original_loan_term",
                                "number_of_borrowers", "seller_name", "servicer_name", "super_conforming_flag"]

    drop_cols: List[str] = ['maturity_date', 'metropolitan_division', 'original_interest_rate', 'property_state',
                            'postal_code', 'mortgage_insurance_percent', 'original_loan_term']

    df = pd.concat([Reducer.simple_ts_split(pd.read_csv(fm_root + 'historical_data_2009Q1.txt', sep='|', index_col=False,
                                                            names=origination_cols, nrows=1_000_000).merge(
            pd.read_pickle(fm_root + 'dev_labels.pkl'), on="loan_sequence_number",
            how="inner").merge(
            pd.read_pickle(fm_root + 'dev_reg_labels.pkl'), on="loan_sequence_number",
            how="inner").drop(columns=drop_cols), sort_key='first_payment_date', split_ratio=[0.8, 0.1, 0.1])])
    return df

class Reducer:

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
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
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

    @staticmethod
    def random_split(df: DataFrame, split_ratio: float = .8):
        train, test = train_test_split(df, test_size=1 - split_ratio)
        train['flag'] = 'train'
        test['flag'] = 'test'
        df = pd.concat([train, test])
        return df