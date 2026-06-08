"""Script to go from all Raw dFAD data to standard form of dataset with cleaned data"""

from CombineSatlink import CombineSatlink
from  CombineMI import CombineMI
from  Combine_SAT_MI import Combine_SAT_MI
from  Cleaning_SAT_MI import Cleaning_SAT_MI

def Combine_clean_alldata():
    CombineSatlink()
    CombineMI()
    Combine_SAT_MI()
    Cleaning_SAT_MI()

if __name__ == '__main__': 
    Combine_clean_alldata()