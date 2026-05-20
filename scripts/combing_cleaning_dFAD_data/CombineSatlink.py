import pandas as pd 
import functions.settings as settings 

""" File combines all satlink data withing the folderpath into .csv, to add new data to this add within the satlink subdirectory
the output .csv file has columns BuoyName,Timestamp,Latitude,Longitude,Speed,MinOfTimes,MaxOfTimes
"""
def CombineSatlink():
    folderpath = settings.RAWDATA_DIR / "Satlink"

    csvfiles =  list(folderpath.glob("*.csv"))
    csv1 = csvfiles[0]
    data = pd.read_csv(csv1,  encoding='utf-16', sep= ";", index_col=None)
    data = data.drop(index = 0)
    ##print(data.head(2))
    print(data["Latitude"].isna().sum())

    ## add All the files with the common formating. 
    for csv in csvfiles[1:]:
        #print(f"Filename :{csv}")
        print(csv)
        try:
            data2 = pd.read_csv(csv,  encoding='utf-16', sep= ";", index_col=None,low_memory=False)
            data2 = data2.drop(index = 0)
        except: 
            data2 = pd.read_csv(csv, sep= ",", index_col=None,low_memory=False)
            data2 = data2.drop(index = 0)
            data2["Timestamp"] = pd.to_datetime(data2["Timestamp"], format = r'%m/%d/%y %H:%M')
            data2["Timestamp"] = data2["Timestamp"].dt.strftime(r'%Y-%m-%d %H:%M')
            print(data2.head(2))
        #print(data2.head(2))
        data2["Timestamp"] = pd.to_datetime(data2["Timestamp"], format = r'%Y-%m-%d %H:%M')
        data = pd.concat([data,data2], ignore_index=True)
        #print(data.shape)
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], format = r'%Y-%m-%d %H:%M')
        print(data2["Timestamp"].min(), data2["Timestamp"].max())


    ## adding non common formating the the dataset. 
    csv = folderpath / "Other Data/2021-202312 Palmyra Export.csv"
    data2 = pd.read_csv(csv, sep= ",", index_col=None,low_memory=False)
    data = pd.concat([data,data2], ignore_index=True)

    csv = folderpath / "Other Data/210925 DR+ES Dump.csv"
    data2 = pd.read_csv(csv, sep= ";", index_col=None,low_memory=False)
    data2 = data2.rename(columns={"Name":"BuoyName"})
    data2["Timestamp"] = pd.to_datetime(data2["StoredTime"], format = r'%Y/%m/%d %H:%M:%S')
    data2["Timestamp"] = data2["Timestamp"].dt.strftime(r'%Y-%m-%d %H:%M')
    print(data2.columns)
    data = pd.concat([data,data2], ignore_index=True)

    data["Timestamp"] = pd.to_datetime(data["Timestamp"], format = r'%Y-%m-%d %H:%M')
    cols_to_keep=['BuoyName', 'Timestamp', 'Latitude', 'Longitude', 'Speed']
    data=data[cols_to_keep]
    data = data.dropna(subset = ["Latitude"])
    print(data.head(2))
    ## Sorting Dataframe
    data = data.sort_values("Timestamp")
    data["MinOfTimes"] = data.groupby(["BuoyName"])['Timestamp'].transform("min")
    data["MaxOfTimes"] = data.groupby(["BuoyName"])['Timestamp'].transform("max")
    data = data.sort_values(by = ['MinOfTimes', "Timestamp"])
    data = data.drop_duplicates()
    print(data.shape)
    print(f"Number of Unique dFADS : {len(data["BuoyName"].unique())}")
    data.to_csv(settings.DATA_DIR / r"Sat_FAD.csv")

if __name__ == '__main__': 
    CombineSatlink()