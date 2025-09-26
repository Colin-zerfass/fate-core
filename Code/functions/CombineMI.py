import pandas as pd 
import glob 
import os 

folderpath = r"C:\FATE\Code\Data\Palmyra_Data_CSV\MI"
csvfiles = glob.glob(os.path.join(folderpath,"*.csv"))
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
data.to_csv(r"C:\FATE\Code\Data\Palmyra Data\MI_FAD.csv")

## Combining git Satlink data
data2 = pd.read_csv(r"C:\FATE\Code\Data\Palmyra Data\Satlink_FAD_Time_21_25.csv")
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
data.to_csv(r"C:\FATE\Code\Data\Palmyra Data\SAT_MI_FAD.csv")
