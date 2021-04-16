import pandas as pd
import numpy as np
import os
from numba import njit, prange, threading_layer, config
import numba
from timeit import default_timer as timer

#filename = 'CSV/Pice_Mar.csv'
directory = "CSV/Final_CSV"
output = "CSV"
config.THREADING_LAYER = 'omp'
num_split = 0

def to_csv():
    print("Started Reading")
    i = 0
    array = os.listdir(directory)
    print("Arr: ", array)
    csv_arr = [x for x in array if '.csv' in x]
    name_arr = []
    for i in csv_arr:
        temp = i.split("_")
        name_arr.append(temp[0])
    uniqueNames = set(name_arr)
    uniqueNames = sorted(uniqueNames)

    for names in uniqueNames:
        frames = []

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if names in filename:
                print(filename)
                #for i in range(num_split):
                    #new_name = name_arr[0] + "_" + name_arr[1] + "_" + str(i) + os.path.splitext(filename)[1]
                    #print(new_name)
                df = pd.read_csv(os.path.join(directory, filename))
                frames.append(df)
        result = pd.concat(frames)
        result.to_csv(os.path.join(output, names + ".csv" ), index=False)
        print("That's all")
    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    to_csv()
