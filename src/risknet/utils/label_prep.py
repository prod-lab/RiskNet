import pandas as pd
from pandas import DataFrame

import numpy as np

from typing import List, Dict, Tuple
import pickle
import logging
logger = logging.getLogger("freelunch")

def label_proc(fm_root, label_sets):

    performance_cols: List[str] = ["loan_sequence_number", "monthly_reporting_period", "current_actual_upb",
                                "current_loan_delinquency_status", "loan_age",
                                "remaining_months_to_maturity",
                                "repurchase_flag", "modification_flag", "zero_balance_code",
                                "zero_balance_effective_date", "current_interest_rate",
                                "current_deferred_upb",
                                "due_date_last_installment",
                                "insurance_recoveries", "net_sales_proceeds", "non_insurance_recoveries",
                                "expenses",
                                "legal_costs", "maintenance_costs", "taxes_and_insurance", "misc_expenses",
                                "actual_loss", "modification_cost", "step_modification_flag",
                                "deferred_payment_modification", "loan_to_value", "zero_balance_removal_upb",
                                "delinquent_accrued_interest"]
    
    for i in label_sets:
        performance_df: DataFrame = pd.read_csv(fm_root + i[0], sep='|', index_col=False,
                                            names=performance_cols, nrows=10_000_000).loc[:,
                                    ["loan_sequence_number", "monthly_reporting_period",
                                    "current_loan_delinquency_status",
                                    "zero_balance_code", "loan_age", "remaining_months_to_maturity"]]
                                    #EC: Added nrows to make faster

        performance_df.loc[:, ["current_loan_delinquency_status", "zero_balance_code"]] = performance_df.loc[:, [
                                                                                "current_loan_delinquency_status",
                                                                                "zero_balance_code"]].astype(str)

        performance_df['default'] = np.where(
            ~performance_df['current_loan_delinquency_status'].isin(["XX", "0", "1", "2", "R", "   "]) |
            performance_df[
                'zero_balance_code'].isin(['3.0', '9.0', '6.0']), 1, 0)

        performance_df['progress'] = performance_df["loan_age"] / (performance_df["loan_age"] + performance_df["remaining_months_to_maturity"])

        performance_df = performance_df.sort_values(['loan_sequence_number', 'monthly_reporting_period'],
                                                    ascending=True).groupby('loan_sequence_number').head(60)

        flagged_loans: DataFrame = pd.DataFrame(performance_df.groupby("loan_sequence_number")['default'].max()).reset_index()

        with open(fm_root + i[1], 'wb') as f:
            pickle.dump(flagged_loans, f)

        regression_flagged_loans: DataFrame = pd.DataFrame(performance_df.loc[performance_df['default'] == 0].groupby("loan_sequence_number")['progress'].max()).reset_index().rename(columns={'progress':'undefaulted_progress'})

        with open(fm_root + i[2], 'wb') as f:
            pickle.dump(regression_flagged_loans, f)

            