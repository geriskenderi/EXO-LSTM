import datetime
import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def dateindex_from_weeknum(weeknum, year):
    """Build datetime index in order to transform data into time series"""

    if weeknum == 1:
        date_str = "" + str(year) + "-01-01"
        dateindex = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    else:
        date_str = "" + str(year) + "-W" + str(weeknum) + "-1"
        dateindex = datetime.datetime.strptime(date_str, "%Y-W%W-%w")

    return dateindex

def dateindex_from_timestamp(timestamp):
    """Build datetime index in order to transform data into time series"""

    dateindex = datetime.datetime.strptime(timestamp, "%Y-%m-%d")

    return dateindex

def ts_train_test_split(df, test_length=0.2):
    # Perform train-test split
    split_index = int(df.shape[0]*test_length)
    train = df[:-split_index]
    test = df[-split_index:]    
    return train, test

def ts_train_test_keepnr(df, keepnr=7):
    # Keep keepnr from the end of the time series as test
    train = df[:-keepnr]
    test = df[-keepnr:] 
    return train, test

# def lagged_corr(datax, datay, lag=0):
#     return datax.corr(datay.shift(lag), method="spearman") # no p-values

# def lagged_corr_matrix(df, lag=0):
#     df = df.dropna()._get_numeric_data()
#     dfcols = pd.DataFrame(columns=df.columns)
#     corr_values = dfcols.transpose().join(dfcols, how='outer')
#     for r in df.columns:
#         for c in df.columns:
#             corr_values[r][c] = lagged_corr(df[r], df[c], lag)
#     return corr_values.astype('float64')

# def calculate_pvalues(df, lag=0):
#     df = df.dropna()._get_numeric_data()
#     dfcols = pd.DataFrame(columns=df.columns)
#     pvalues = dfcols.transpose().join(dfcols, how='outer')
#     for r in df.columns:
#         for c in df.columns:
#             pvalues[r][c] = round(spearmanr(df[r],df[c].shift(lag, fill_value=0))[1], 4) #CONSULT: fill_value
#     return pvalues.astype('float64')

# def cols_with_most_corr(df, forecast_target_col=0, corr_threshhold=0.10):
#     corr_matrix = df.corr(method="spearman")
#     corrmat_row = corr_matrix.iloc[forecast_target_col, :]
#     corrmat_row_vals = corrmat_row.values.tolist()
#     # Get only the columns that have a correlation coefficient >= corr_threshhold or =< -corr_threshhold
#     correlated_features = []
#     for val in corrmat_row_vals:
#         if (val > corr_threshhold or val < -corr_threshhold):
#             pct_zeros_in_col = df[df.iloc[:, corrmat_row_vals.index(val)] == 0].count(axis=0)[0]/len(df.index)
#             if pct_zeros_in_col < 0.5:
#                 correlated_features.append(corrmat_row_vals.index(val))

#     return correlated_features

# def cols_with_most_lagged_corr(df, lag=-1, forecast_target_col=0, corr_threshhold=0.10):
#     corr_matrix = lagged_corr_matrix(df, lag=lag)
#     corrmat_row = corr_matrix.iloc[forecast_target_col, :]
#     corrmat_row_vals = corrmat_row.values.tolist()
#     # Get only the columns that have a correlation coefficient >= corr_threshhold or =< -corr_threshhold
#     correlated_features = []
#     for val in corrmat_row_vals:
#         if (val > corr_threshhold or val < -corr_threshhold):
#             pct_zeros_in_col = df[df.iloc[:, corrmat_row_vals.index(val)] == 0].count(axis=0)[0]/len(df.index)
#             if pct_zeros_in_col < 0.5:
#                 correlated_features.append(corrmat_row_vals.index(val))

#     return correlated_features



# def cols_with_sig_pval(df, forecast_target_col=0, sig_lvl=0.05, lag=0):
#     pval_matrix = calculate_pvalues(df, lag)
#     corrmat_row = pval_matrix.iloc[forecast_target_col, :]
#     corrmat_row_vals = corrmat_row.values.tolist()
#     print(corrmat_row_vals)
#     # Get only the columns that have statistically significant correlation
#     correlated_features = []
#     for idx, val in enumerate(corrmat_row_vals):
#         if (val < sig_lvl): # The p-value given is for a two-sided test
#             pct_zeros_in_col = df[df.iloc[:, idx] == 0].count(axis=0)[0]/len(df.index)
#             if pct_zeros_in_col < 0.5:
#                 correlated_features.append(idx)

#     return correlated_features

# def corr_feature_selection(df, forecast_target_col=0, min_threshhold=0.10, sig_lvl=0.05, lag=0):
#     pval_matrix = calculate_pvalues(df, lag)
#     corr_matrix = lagged_corr_matrix(df, lag)
#     corrmat_row = corr_matrix.iloc[forecast_target_col, :]
#     pval_row = pval_matrix.iloc[forecast_target_col, :]
#     corrmat_row_vals = corrmat_row.values.tolist()
#     pval_row_vals = pval_row.values.tolist()

#     # Get only the columns that have statistically significant correlation
#     correlated_features = []
#     for idx, val in enumerate(corrmat_row_vals):
#         if ((val < -min_threshhold or val > min_threshhold) and pval_row_vals[idx] < sig_lvl):
#             pct_zeros_in_col = df[df.iloc[:, idx] == 0].count(axis=0)[0]/len(df.index)
#             if pct_zeros_in_col < 0.5:
#                 correlated_features.append(idx)

#     return correlated_features