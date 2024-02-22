import pandas as pd
import numpy as np
import dask.dataframe as dd
import dask.array as da

def parquet_convert(data1, data2):
    fm_root = "/home/zahmed/private/RiskNet/src/risknet/data/"
    monthly = dd.read_csv(fm_root+data1, sep='|', header = None,dtype={23: 'object',
           24: 'object',
           28: 'object',
           29: 'object',
           3: 'object',
           7: 'object'})

    monthly.columns = ["loan_sequence_number", "monthly_reporting_period", "current_actual_upb",
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
                                       "delinquent_accrued_interest","del_disaster","borrower_assistance","month_mod_cost","interest_bearing"]

    monthly['row_hash'] = monthly.assign(partition_count=50).partition_count.cumsum() % 50

    monthly.to_parquet("/home/zahmed/private/RiskNet/src/risknet/data/" + '/monthly.parquet', partition_on = "row_hash")

    org = dd.read_csv(fm_root+data2, sep='|', header = None,dtype={25: 'object',
           26: 'object',
           28: 'object'})
    org = org.drop(columns = {26, 27,28,29,30,31})
    org.columns = ["credit_score", "first_payment_date", "first_time_homebuyer", "maturity_date",
                                           "metropolitan_division", "mortgage_insurance_percent", "number_of_units",
                                           "occupancy_status", "orig_combined_loan_to_value", "dti_ratio",
                                           "original_unpaid_principal_balance", "original_ltv", "original_interest_rate",
                                           "channel", "prepayment_penalty_mortgage", "product_type", "property_state",
                                           "property_type", "postal_code", "loan_sequence_number", "loan_purpose",
                                           "original_loan_term",
                                           "number_of_borrowers", "seller_name", "servicer_name", "super_conforming_flag"]
    org['row_hash'] = org.assign(partition_count=50).partition_count.cumsum() % 50

    org.to_parquet("/home/zahmed/private/RiskNet/src/risknet/data/" + '/org.parquet', partition_on = "row_hash")
    
    print("done")

#parquet_convert('historical_data_time_2009Q1.txt','historical_data_2009Q1.txt')
