import pandas as pd
import numpy as np
# Extract an index for each day of the year from dates
def getSeasonalCycle(y, dates):
    dates = pd.DatetimeIndex(dates)
    dayofyear = dates.dayofyear

    seasonal_cycle = np.zeros(shape = y.shape)

    # Compute the mean for each day from y:
    for iday in range(seasonal_cycle.shape[0]):
        seasonal_cycle[iday, ...] = y[dayofyear == dayofyear[iday], ...].mean(axis = 0)
        
    return seasonal_cycle