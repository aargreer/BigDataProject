import pandas
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
        filename = os.fsdecode(file)
        if filename.endswith("_0.csv"):
            print(filename)
            start = timer()
            prev = 0
            Totaldf = pandas.DataFrame()
            frames = []
            new_name = os.path.splitext(filename)[0]  # + "_" + str(i) + os.path.splitext(filename)[1]
            name_arr = new_name.split("_")
            for i in range(num_split):

                new_name = name_arr[0] + "_" + name_arr[1] + "_" + str(i) + os.path.splitext(filename)[1]
                print(new_name)
                df = pd.read_csv(os.path.join(directory, new_name))
                frames.append(df)
                #new_df.to_csv(new_name, index=False)
            result = pd.concat(frames)
            result.to_csv(name_arr[0] + "_" + name_arr[1] + os.path.splitext(filename)[1], index=False)
            end = timer()
            print("Time taken: ", (end - start))
            print(i)
            i += 1
    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    to_csv()
