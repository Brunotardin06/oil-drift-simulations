import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as crt
from scipy import stats
import os
import io
from datetime import datetime


def loadData(fileReference, fileComparison, dateTimeRef):
    reference = xr.open_dataset(fileReference)
    comparison = xr.open_dataset(fileComparison)

    referenceSlice = reference.sel(time=dateTimeRef, method="nearest")



