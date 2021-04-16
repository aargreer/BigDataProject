import random

import pandas as pd
import numpy as np
import math
import os
import datetime
from timeit import default_timer as timer

#filename = 'CSV/Pice_Mar.csv'
pd.set_option('display.max_columns', None)
pd.set_option("display.precision", 13)
year = 2015

def to_csv():
    print("Started Reading")
    i = 0
    filename = "firedata_" + str(year) + ".csv"
    print(filename)
    print("Reading Dataframe")
    df = pd.read_csv(filename, low_memory=False)
    df.astype({'NFIREID' : 'int32'}).dtypes
    unique = df['NFIREID'].unique()
    frames = []
    i = 0
    for name in unique:
        temp = df.query('NFIREID == ' + str(name)).reset_index()
        frames.append(temp.iloc[np.argmin(abs(temp['Y'] - np.median(temp['Y']))), :])
        i += 1

    print(i)
    conc_df = pd.DataFrame(frames)
    conc_df = conc_df.reset_index()
    conc_df = conc_df.drop('index', axis=1)
    conc_df = conc_df.drop('level_0', axis=1)
    df = conc_df.sort_values(by=['NFIREID'])

    #Drop fires with no date (Can't use these)
    df = df[df['SDATE'].notna() | df['EDATE'].notna() | df['AFSDATE'].notna() ]
    # Nasty lamda to impute start date if not there
    df['SDATE'] = df.apply(lambda row:
                           (row['AFSDATE']
                           if row['EDATE'] == np.nan
                              or (row['EDATE'] is None)
                              or (pd.isnull(row['EDATE']))
                           else row['EDATE'])
                           if (row['SDATE'] == np.nan)
                              or (row['SDATE'] is None)
                              or (pd.isnull(row['SDATE']))
                           else row['SDATE']
                           , axis=1)

    # Add negative cases (Spots without fires)
    neg = pd.read_csv("Tree_Data.csv")
    neg.drop(['Genus', 'Percent'], inplace=True, axis=1)
    neg.sort_index(inplace=True)
    print("Length: ", len(neg.index))
    nul = {'X': [np.nan], 'Y': [np.nan]}
    nul = pd.DataFrame(data=nul)
    selection = (neg.iloc[np.random.choice(np.arange(len(neg)), 10000, False)]).copy()
    selection['X'] = selection.apply(
        lambda row: row['X']
                    if np.argmin(abs(df['X'].round(2) - row['X']) + abs(df['Y'].round(2) - row['Y'])  ) > 1
                    else np.nan,
        axis=1)
    selection['Y'] = selection.apply(
        lambda row: row['Y']
        if np.argmin(abs(df['X'].round(2) - row['X']) + abs(df['Y'].round(2) - row['Y'])) > 1
        else np.nan,
        axis=1)

    selection = selection[selection['X'].notna()]
    #print("Neg: ", neg['X'], "Neg Y: ", neg['Y'])

    dates = []
    for i in range(len(selection)):
        dates.append((datetime.date(year, 1, 1) + datetime.timedelta(
        random.randrange((datetime.date(year, 12, 31) - datetime.date(year, 1, 1)).days))).strftime("%Y/%m/%d"))
    selection['SDATE'] = dates
    selection['BURNCLAS'] = 0
    selection = (selection.iloc[np.random.choice(np.arange(len(selection)), len(df), False)]).copy()
    print(selection.head())
    final_df = pd.concat([df, selection])
    #write result to csv
    final_df.to_csv("fire_processed_" + str(year) + ".csv", columns=["X", "Y", "SDATE", "BURNCLAS"] ,index=False)
    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    to_csv()
