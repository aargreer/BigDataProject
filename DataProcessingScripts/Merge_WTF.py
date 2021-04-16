import pandas as pd
import numpy as np
from datetime import datetime
import os
from timeit import default_timer as timer

#filename = 'CSV/Pice_Mar.csv'
directory = "Finished"
year = 2015
def to_csv():
    print("Started Reading")
    i = 0
    fire    = "fire_processed_" + str(year) + ".csv"
    climate = "climate_summaries_" + str(year) + ".csv"
    tree    = "Tree_Data.csv"
    fire_df = pd.read_csv(fire, parse_dates= ["SDATE"])
    climate_df = pd.read_csv(climate, parse_dates= ["Date"])
    tree_df = pd.read_csv(tree)
    print(tree_df.columns.tolist())
    print(fire_df.columns.tolist())
    # Assign each fire a tree type

    fire_df['tree_genus'] = fire_df.apply(lambda row: tree_df.iloc[np.argmin(abs(tree_df['X'] - row['X']) + abs(tree_df['Y'] - row['Y'])), :]['Genus'],  axis=1)
    #print(fire_df.head())

    # match weather data to fire data using date and coordinates
    fire_df['SDATE'] = pd.to_datetime(fire_df['SDATE'].dt.strftime('%Y-%m'), format="%Y-%m")
    climate_df['Date'] = pd.to_datetime(climate_df['Date'].dt.strftime('%Y-%m'), format="%Y-%m")
    fire_climate = fire_df.apply(lambda row: climate_df.iloc[
                                             np.argmin(abs(climate_df['Long'] - row['X'])
                                             + abs(climate_df['Lat'] - row['Y'])
                                             + abs( ((climate_df['Date'] - row['SDATE']).dt.days )) * 1000 )
                                             , :],  axis=1)

    bigdata = pd.concat([fire_df, fire_climate], ignore_index=False, sort=False, axis=1)

    print(bigdata.head())
    bigdata.to_csv(os.path.join(directory, "fire_data_" + str(year) + ".csv"), columns=["X", "Y", "Tm", "P", "tree_genus", "BURNCLAS"], index=False)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    to_csv()