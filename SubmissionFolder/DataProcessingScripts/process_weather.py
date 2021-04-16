import pandas
import pandas as pd
import numpy as np
import datetime as dt
import os
import math
from timeit import default_timer as timer

#filename = 'CSV/Pice_Mar.csv'

directory = '2015'
output = "."
num_split = 15

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
    i = 1
    for names in uniqueNames:
        frames = []
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if names in filename:
                print(filename)
                df = pd.read_csv(os.path.join(directory, filename))
                df["Date"] = dt.date(int(directory), int(filename.split("_")[4].split("-")[0]), 5)
                noNan = df[df['Tm'].notna()]
                noNan = noNan[noNan['Tm'].notnull()]

                # Massive awful lamda statement to impute Tm (mean temperature)
                df['Tm'] = df.apply(lambda row : noNan.iloc[np.argmin(abs(noNan['Long'] - row['Long'] ) + abs(noNan['Lat'] - row['Lat'] ) ), :]['Tm'] if (row['Tm'] == np.nan) or (row['Tm'] is None) or (math.isnan(row['Tm'])) else row['Tm'], axis=1)
                df['P'] = pd.to_numeric(df['P'], errors='coerce')
                #df.dropna(inplace=True)
                noNan = df[df['P'].notna()]
                noNan = noNan[noNan['P'].notnull()]

                # Massive awful lamda statement to impute P (Precipitation(ml))
                df['P'] = df.apply(lambda row: noNan.iloc[np.argmin(
                    abs(noNan['Long'] - row['Long']) + abs(noNan['Lat'] - row['Lat'])), :]['P']
                    if  (row['P'] == np.nan)
                        or (row['P'] is None)
                        or (math.isnan(row['P'])) else row['P'], axis=1)
                #print(new_df['Tm'].head(25))
                # Impute using closest weather station

                # append imputed df
                frames.append(df)

        result = pd.concat(frames)
        result.to_csv(os.path.join(output, "climate_summaries_" + directory + ".csv" ), columns=["Long", "Lat", "Tm", "P", "Date"], index=False)
        print("That's all")
    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    to_csv()