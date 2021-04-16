import pandas as pd
import numpy as np
import os
from numba import njit, prange, threading_layer, config
import numba
from timeit import default_timer as timer

#filename = 'CSV/Pice_Mar.csv'
directory = "CSV/Final_CSV"
config.THREADING_LAYER = 'omp'

def to_csv():
    print("Started Reading")
    i = 0
    frames = []
    for file in os.listdir(directory):
        print("Here")
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, filename))
            #df['Genus'] = os.path.splitext(filename)[0]
            df.insert(3, 'Genus', os.path.splitext(filename)[0], allow_duplicates=True)
            frames.append(df)

    result = pd.concat(frames)
    #print("Original", result.isnull().sum(), "\n", result.head())

    print("Total number: ", len(result.index))
    print("Begin Grouping")
    #pairs = result.groupby(['X', 'Y'])
    #pairs = df.groupby(['X', 'Y'], as_index=False).apply(lambda s: s.loc[s.Percent.idxmax(), ['X', 'Y', 'Percent', 'Genus']]).reset_index(drop=True)
    #pairs = result.groupby(['X', 'Y']).agg({'Percent': ['max'], 'Genus' : ['first']}).reset_index()
    #pairs = result.groupby(['X', 'Y'], as_index=False)
    #print(len(pairs))
    indx = result.groupby(['X', 'Y'])['Percent'].transform(max) == result['Percent']
    pairs = result[indx]
    print("length: ", len(pairs.index))
    print(pairs.head())
    pairs.to_csv(os.path.join(directory, "Tree_Data.csv"), index=False)
    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    to_csv()
