import pandas as pd
import numpy as np
import os
from numba import njit, prange, threading_layer, config
import numba
from timeit import default_timer as timer

#filename = 'CSV/Pice_Mar.csv'
directory = "CSV"
config.THREADING_LAYER = 'omp'

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
            df = df.round(2)
            print("Start")
            pairs = df.groupby(['X','Y']).size().reset_index().rename(columns={0:'count'})
            print("Number of pairs: ", len(pairs.index))
            start = timer()
            testArray = lamda_func(df.to_numpy(), pairs['X'].to_numpy(), pairs['Y'].to_numpy(), len(df.index), (len(pairs.index)), np.zeros((len(pairs.index), 3), dtype=float))
            end = timer()
            print("Time taken: ", (end - start))
            correctedDF = pd.DataFrame(columns=["X", "Y", "Percent"], data= testArray)
            correctedDF = correctedDF.round(2)
            correctedDF.to_csv(os.path.join("CSV/Final_CSV", filename), index=False)
        print(i)
        i += 1
    return

@njit(nopython=True, parallel=True)
def lamda_func(df, rowx, rowy, lendf, leny, sum):
    for i in range(leny):
        x = 0
        y = 0
        v = 0
        len = 0
        #print(i)
        for j in prange(lendf):
            if (df[j, numba.i8(0)] == rowx[i] and df[j, numba.i8(1)] == rowy[i]):
                x += df[j, numba.i8(0)]
                y += df[j, numba.i8(1)]
                v += df[j, numba.i8(2)]
                len += 1

        #mean = [x/len(x), y/len(y),v/len(v)]
        sum[i][0] = x / len
        sum[i][1] = y / len
        sum[i][2] = v / len
    return sum

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    to_csv()
