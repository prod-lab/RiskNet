import dask.dataframe as dd
import pickle
performance_cols = [
        "loan_sequence_number", "monthly_reporting_period", "current_actual_upb",
        "current_loan_delinquency_status", "loan_age", "remaining_months_to_maturity",
        "repurchase_flag", "modification_flag", "zero_balance_code",
        "zero_balance_effective_date", "current_interest_rate", "current_deferred_upb",
        "due_date_last_installment", "insurance_recoveries", "net_sales_proceeds",
        "non_insurance_recoveries", "expenses", "legal_costs", "maintenance_costs",
        "taxes_and_insurance", "misc_expenses", "actual_loss", "modification_cost",
        "step_modification_flag", "deferred_payment_modification", "loan_to_value",
        "zero_balance_removal_upb", "delinquent_accrued_interest"
    ]

origination_cols = ["credit_score", "first_payment_date", "first_time_homebuyer", "maturity_date",
                                "metropolitan_division", "mortgage_insurance_percent", "number_of_units",
                                "occupancy_status", "orig_combined_loan_to_value", "dti_ratio",
                                "original_unpaid_principal_balance", "original_ltv", "original_interest_rate",
                                "channel", "prepayment_penalty_mortgage", "product_type", "property_state",
                                "property_type", "postal_code", "loan_sequence_number", "loan_purpose",
                                "original_loan_term",
                                "number_of_borrowers", "seller_name", "servicer_name", "super_conforming_flag"]

def get_default(data, col_1, col_2):
    if (data[col_1] in ["XX", "0", "1", "2", "R", "   "]) or (data[col_2] in ['3.0', '9.0', '6.0']):
        return 1
    else:
        return 0
    
def add_progress(data, col_1, col_2):
    return data[col_1] / (data[col_1] + data[col_2])
    
def data_processing(data):
    table = dd.read_csv('historical_data_time_2009Q1.txt', sep="|", header=None, dtype='object') #, convert_options=convert_options)
    cols = list(table.columns[:28])
    table = table[cols]
    reassign: dict = dict()
    for i in range(0, 28):
        reassign[i] = performance_cols[i]
    table = table.rename(columns=reassign)
    table['loan_sequence_number'] = table['loan_sequence_number'].astype(str)
    table['zero_balance_code'] = table['zero_balance_code'].astype(str)
    table['loan_age'] = table['loan_age'].astype(float)
    table['remaining_months_to_maturity'] = table['remaining_months_to_maturity'].astype(float)
    default_series = table.apply(
    get_default,
    args=('current_loan_delinquency_status', 'zero_balance_code'),
    axis=1,
    meta=('default', int)
)
    table['default'] = default_series   
    progress_series = table.apply(
    add_progress,
    args=('loan_age', 'remaining_months_to_maturity'),
    axis=1,
    meta=('progress', float)
)

    table['progress'] = progress_series
    flagged_loans = table.groupby("loan_sequence_number")['default'].max().reset_index()
    regression_flagged_loans = table.loc[table['default'] == 0].groupby("loan_sequence_number")['progress'].max().reset_index().rename(columns={'progress':'undefaulted_progress'})
    with open( 'dev_labels.pkl', 'wb') as f:
        pickle.dump(flagged_loans, f)
    with open('dev_reg_labels.pkl', 'wb') as f:
        pickle.dump(regression_flagged_loans, f)
    
    hist = dd.read_csv('historical_data_2009Q1.txt', sep="|", header=None, dtype='object')


    cols = list(hist.columns[:26])
    hist = hist[cols]
    reassign: dict = dict()
    for i in range(0, 26):
        reassign[cols[i]] = origination_cols[i]

    hist = hist.rename(columns=reassign)