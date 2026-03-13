import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as crt
from scipy import stats
import os
import io
from datetime import datetime


def loadData(fileReference, fileComparison, dateTimeRef, dateTimeStartComparison):
    reference = xr.open_dataset(fileReference)
    comparison = xr.open_dataset(fileComparison)

    referenceSlice = reference.sel(time=dateTimeRef, method="nearest")
    comparisonSlice = comparison.sel(time=dateTimeStartComparison, method="nearest")
    print(referenceSlice)
    print("="*60)
    print(comparisonSlice)


def main():
    file_ref = 'analysis\cmems_obs-mob_glo_phy-cur_my_0.25deg_PT1H-i_1771818740018.nc'
    file_cmp = 'analysis\cmems_obs-mob_glo_phy-cur_nrt_0.25deg_PT1H-i_1771818690342.nc'
    datetime_str = "2026-06-12T00:00:00"
    loadData(file_ref, file_cmp, datetime_str, datetime_str)



if __name__ == "__main__":
    main()




    



