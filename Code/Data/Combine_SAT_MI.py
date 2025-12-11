""" Combines the .csv files produced by CombineMI.py and Combine_SAT.py"""

import pandas as pd
## Combining git Satlink data
data = pd.read_csv(r"MI_FAD.csv")
data2 = pd.read_csv(r"Sat_FAD.csv")
data2["Timestamp"] = pd.to_datetime(data2["Timestamp"], format = r'%Y-%m-%d %H:%M:%S')

data = pd.concat([data, data2])
data["Timestamp"] = pd.to_datetime(data["Timestamp"], format = r'%Y-%m-%d %H:%M:%S')
data = data.sort_values("Timestamp")
data = data.sort_values(by = ['MinOfTimes', "Timestamp"])
data = data.drop_duplicates()
data = data.drop(columns='Unnamed: 0')
print(data.columns)
print(data.shape)
print(f"Number of Unique dFADS COMBINED: {len(data["BuoyName"].unique())}")
data.to_csv(r"SAT_MI_FAD.csv")