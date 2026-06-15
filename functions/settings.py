## points outside of the proejct to specific file locations 
## inside core/Functions
from pathlib import Path

FUNCTIONS_DIR = Path(__file__).resolve().parent
CORE_DIR = FUNCTIONS_DIR.parent
ROOT_DIR = CORE_DIR.parent # the DIR that is right outside of CORE


### _____________
### DATA LOCTIONS
###______________
DATA_DIR = ROOT_DIR / 'Data' # Parent DIR of all Data

#dFAD data
RAWDATA_DIR =   DATA_DIR / 'Palmyra_Data_CSV' ## Location of Raw dFAD data (colection of qunaterly data download sets)
dFAD_DATA_UNMAPPED = DATA_DIR / "SAT_MI_FAD_cleanedspeeds_2026-01-01.parquet"
dFAD_DATA =     DATA_DIR / 'SAT_MI_FAD_cleanedspeeds_2026-01-01_mapped_all.parquet'

#Drifter Data
RAWDRIFTER_DIR = DATA_DIR / 'Drifters'
DRIFTER_DATA = DATA_DIR / 'Drifter_cleaned_2026_06.parquet'
DRIFTER_GEOFENCED_DATA_UNMAPPED = DATA_DIR / 'Drifter_cleaned_geofenced_2026_01.parquet'
DRIFTER_GEOFENCED_DATA = DATA_DIR / 'Drifter_cleaned_geofenced_mapped_2026_01.parquet'

#model data
GLORYS_FILE  = DATA_DIR / 'cmems_2021_2026.nc'
OSCAR_FILE  = DATA_DIR / 'OSCAR_combined_2021_2026.nc'
ERA5_FILE   = DATA_DIR / 'ERA5_10m_winds_Stokes_2021_2026.nc' 

# extra 
BATHYMETRY_FILE = DATA_DIR / 'bath.nc'


####_________________
### FIGURE FILE PATHS
###___________________
#making Figures DIR if doesn't exist 
__figures = 'Figures'
if not (ROOT_DIR / __figures).is_dir():
    (ROOT_DIR / __figures).mkdir()

FIGURES_DIR = ROOT_DIR / __figures

__paper = 'Paper_test'
if not (FIGURES_DIR / __paper).is_dir(): # make DIR to put Paper figures into 
    (FIGURES_DIR/ __paper).mkdir()
    print(f'Made DIR {FIGURES_DIR / __paper}')

FIGURES_PAPER_DIR = FIGURES_DIR / __paper


##_______
### Parcels
###_______ 

LOG_FILE  = CORE_DIR / 'scripts' / 'forecasts' / 'output' / 'Output_log.txt'

__forecast_outputdir  = 'Forecast_data'
if not (DATA_DIR / __forecast_outputdir).is_dir(): #make DIR for forecast data 
    (DATA_DIR/ __forecast_outputdir).mkdir()
    print(f'Make DIR {DATA_DIR/ __forecast_outputdir}')

FORECAST_DIR = DATA_DIR / __forecast_outputdir
