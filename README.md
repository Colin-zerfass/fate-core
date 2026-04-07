# FATE NASA

### Table of Contents
- [Project Overview](#project-overview)
- [Creating an Environment and setup](#creating-an-environment-and-setup)
- [Master dFAD datset format](#dfad-dataset)
- [Handling new dFAD data  ](#handling-new-dfad-data)
  - [Adding and Cleaning Data](#adding-and-cleaning-new-data)
- [Analysis](#analysis)
- [ocean parcels kriging and cmems](#ocean-parcels-kriging-and-cmems)

This is a test change
  
# Project Overview 
This repository has code for the analysis and forcasting of dFADs. This work was part of a NASA OTEM internship working on the FATE project. Inside are scripts to combine, clean, and add new dFAD data into a standard form. Also a bunch of analaysis of the dFAD data, spacial distibutions, speeds and compairson to CMEMS (a dynamical Global ocean assimulated model). Also included are methods to forcast dFADs based of CMEMS, Krigging, and persistence. 

A package of plotting and analysis tools has been developed to aid future work  in `Code/functions` 

# Creating an Environment and setup
stating the environment: 

    python -m venv .venv
Then Activate enviroment and install requirments

    pip install -r requirements.txt

 Installing the package developed(in editing mode) called functions

    
    pip install -e .\Code\
 
 This creates the Package of fucntions to be used throughout the repository in edit mode. This means changes to the utilities in the package are updated on the next `import functions` call. This has `functions.plotting` and `functions.funcs`

# dFAD Dataset
All dFADs data has been combined and cleaned into a single file `Mapped_SAT_MI_Cleanedspeeds.parquet`. This file is simailar to a .csv but is able to store the object type also such as shapely.linestrings and lists not as strings
open with `geopandas.read_parquet()`

| index | BuoyName   | MinOfDate           | geometry | TimeStamp|
|------:|------------|---------------------|----------|----------|
|     5 | M3i+572309 | 2021-06-28 11:12:00 |shapely.linestring| [2021-6-28 11:12:00 , 2021-6-28 14:12:00 ...  ]
|     6 | SLX+584930 | 2024-07-28 8:12:00 |shapely.linestring| [2024-7-28 11:12:00 , 2024-6-28 14:12:00 ...  ]

This dataset has a row for each drifter. Some column have a list of values at each point such as the `TimeStamp` column, which provies the timestamp at each gps location. Similary rows like speed are the calcuated speed for each column. These all have the same legth as number of points which makes it easy for alligning data. 

# Handling new dFAD data 
There are directories already in place to put the dFAD data into. `Code/Data/`. Here is where the master dFAD dataset goes `Mapped_SAT_MI_Cleanedspeeds.parquet` 

The raw dFAD data can be placed in `Palmyra_CSV` Satlink and MI respectivy. This is important to be able to add new data to dataset. 

## Adding and cleaning New Data
For adding new data releases from Satlink or MI (Not the live data, different process for that)

    cd Code/Data

Place new data in Palmyra_Data_CSV along with all of the previous data. 
then: 

    python Combine_clean_alldata.py

This method recombines and cleans all data into the standard form. Although all previous data is not needed, if its not trajectories of dFADs still active at the end of the previvous data dump and at the start of the next datadump will not be merged into one trajectory (shapely.linstring)

The cleaning process is all taken care of as part of `Combine_clean_alldata.py`. To clean high and low speeds have been thrown out and the respective timestamps and gps points have been removed from the lists. keeping the lists the same size as number of points in the linestring. 
# Getting Model data (GLORYs, OSCAR, ERA5)
Follow these Process to download GLORYs, OSCAR and ERA5 wind data. This renames variables in the dataset into a standard form for the codebase. 

### CMEMS (GLORYsV1.2 daily currents data and Monthly means)
The notebook `Code\Data\copernicous_Download.ipynb` to download cmems dataset (this will ask for your compernicous password, need an account with Copernicous )

To combine with an old cmems dataset ie. you already have data from the year 2022 and just downloaded the year 2023 and would like to append them. 

    cd Code/Data
    python Combine_cmems_data.py


### OSCAR downdload
OSCAR downloads one day at a time from the podaac datadownloader and then each day has to be combined into one netCDF file. 

From the main dir (FATE) run in the terminal. Update your envirment path and path to the project. 


    podaac-data-downloader -c OSCAR_L4_OC_INTERIM_V2.0 -d Code\Data\OSCAR -sd 2024-01-01T00:00:00Z -ed 2025-01-01T00:00:00Z -b="-164,4.25,-160.5,8" --process "C:\FATE\.venv\Scripts\python.exe C:\FATE\Code\Data\OSCAR\OSCAR_download_processing.py"

As it downloads the data it combines each file into a .zarr file. Then run this next script to conver to .nc and renames the variables into the same format as cmems. Also at this step you can specify if you want it to merge with a previous dataset. 

    cd Code\Data\OSCAR
    python OSCAR_rename_nc.py

### ERA5 Download 
to download era5 data use `Code\Data\ERA5_download.ipynb`

then to combine new data with and rename ERA5 data 

    cd Code\Data\
    python Combine+process_ERA5.py

# Analysis 
Much of the analysis of dFAD is within the directory `Code`. much of this is not the most organized portion of the codebase. 

# Ocean Parcels Kriging and cmems
The forcasting portion including the analysis of forecasting in done within `Code/Parcels/`. This uses the package ocean parcels. To produce new forcasts and switching between using cmems and krigging field all this is needed to be changed in the field itself. 

