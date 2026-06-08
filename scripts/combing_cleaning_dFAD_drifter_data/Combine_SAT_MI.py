""" Combines the .csv files produced by CombineMI.py and Combine_SAT.py"""

import pandas as pd
import functions.settings as settings
## Combining git Satlink data


def Combine_SAT_MI():
    data = pd.read_csv(settings.DATA_DIR / "MI_FAD.csv")
    data2 = pd.read_csv(settings.DATA_DIR / "Sat_FAD.csv")
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
    data.to_csv(settings.DATA_DIR / "SAT_MI_FAD.csv")

if __name__ == '__main__':
    Combine_SAT_MI()