import pandas as pd 
import functions.settings as settings

"""Combines all Marine Instruments data into one .csv , to add new data place it into the MI subdirectory, 
the final .csv has collumn names 'BuoyName', 'Timestamp', 'Latitude', 'Longitude', 'Speed'"""

def CombineMI():
    folderpath = settings.RAWDATA_DIR / r"MI"
    csvfiles = list(folderpath.glob('*.csv'))
    csv1 = csvfiles[0]
    data = pd.read_csv(csv1, sep= ";", index_col=None)
    data["DATE"] = pd.to_datetime(data["DATE"], format = r'%m/%d/%Y %H:%M')
    data["DATE"] = data["DATE"].dt.strftime(r'%Y-%m-%d %H:%M')


    for csv in csvfiles[1:]:
        print(csv)
        data2 = pd.read_csv(csv, sep= ";", index_col=None,low_memory=False)

        data2["DATE"] = pd.to_datetime(data2["DATE"], format = r'%m/%d/%Y %H:%M')
        data2["DATE"] = data2["DATE"].dt.strftime(r'%Y-%m-%d %H:%M')
        data = pd.concat([data,data2], ignore_index=True)
        #print(data.shape)
        data["DATE"] = pd.to_datetime(data["DATE"], format = r'%Y-%m-%d %H:%M')
        print(data2["DATE"].min(), data2["DATE"].max())

    data = data.rename(columns = {"NAME" :"BuoyName", "LATITUDE":"Latitude", "LONGITUDE":"Longitude", "SPEED": "Speed", "DATE" : "Timestamp"})

    cols_to_keep=['BuoyName', 'Timestamp', 'Latitude', 'Longitude', 'Speed']
    data=data[cols_to_keep]
    data = data.dropna(subset = ["Latitude"])
    data = data.sort_values("Timestamp")
    data["MinOfTimes"] = data.groupby(["BuoyName"])['Timestamp'].transform("min")
    data["MaxOfTimes"] = data.groupby(["BuoyName"])['Timestamp'].transform("max")
    data = data.sort_values(by = ['MinOfTimes', "Timestamp"])
    data = data.drop_duplicates()
    print(data.shape)
    print(f"Number of Unique dFADS : {len(data["BuoyName"].unique())}")
    data.to_csv(settings.DATA_DIR / r"MI_FAD.csv")

if __name__ == '__main__':
    CombineMI()

