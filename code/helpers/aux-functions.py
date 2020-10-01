import datetime
import pandas as pd
import numpy as np

def dateindex_from_weeknum(weeknum, year):
    # Build datetime index by specifying the week number and the year (e.g 12th week of 2016)

    if weeknum == 1:
        date_str = "" + str(year) + "-01-01"
        dateindex = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    else:
        date_str = "" + str(year) + "-W" + str(weeknum) + "-1"
        dateindex = datetime.datetime.strptime(date_str, "%Y-W%W-%w")

    return dateindex

def dateindex_from_timestamp(timestamp):
    # Build datetime index by specifying an actual timestamp(e.g 2016-01-01)

    dateindex = datetime.datetime.strptime(timestamp, "%Y-%m-%d")
    return dateindex

def ts_train_test_split(df, test_length=0.2):
    # Perform train-test split by keeping the specified percentage as test
    split_index = int(df.shape[0]*test_length)
    train = df[:-split_index]
    test = df[-split_index:]    
    return train, test

def ts_train_test_keepnr(df, keepnr=7):
    # Perform train-test split by keeping the specified number as test
    train = df[:-keepnr]
    test = df[-keepnr:] 
    return train, test

def smape(gt, pred):
    return 100 / len(gt) * np.sum(np.abs(pred - gt) / (np.abs(gt) + np.abs(pred)))