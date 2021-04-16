import pandas as pd
import numpy as np
import os
from numba import njit, prange, threading_layer, config
import numba
from timeit import default_timer as timer

#filename = 'CSV/Pice_Mar.csv'
directory = "CSV"
config.THREADING_LAYER = 'omp'
num_split = 15

def to_csv():
    print("Started Reading")
    i = 0
    for file in os.listdir(directory):
        print("Here")
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            print(filename)
            print("Reading Dataframe")
            df = pd.read_csv(os.path.join(directory, filename))
            print("Read dataframe")
            start = timer()
            prev = 0
            for i in range(num_split):
                new_name = os.path.splitext(filename)[0] + "_" + str(i) + os.path.splitext(filename)[1]
                print(new_name)
                index = int(len(df.index) / num_split) * (i + 1)
                new_df = df.iloc[prev:index, :]
                new_df.to_csv(new_name, index=False)
                prev = index
            end = timer()
            print("Time taken: ", (end - start))
        print(i)
        i += 1
    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    to_csv()
