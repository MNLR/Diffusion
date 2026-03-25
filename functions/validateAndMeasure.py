from pickle import FALSE
import numpy as np
import scipy.stats as stats
import pandas as pd

# Extract an index for each day of the year from dates
def getSeasonalCycle(y, dates):
    dates_ = pd.DatetimeIndex(dates)
    dayofyear = dates_.dayofyear

    seasonal_cycle = np.zeros(shape = y.shape)

    # Compute the mean for each day from y:
    for iday in range(seasonal_cycle.shape[0]):
        seasonal_cycle[iday, ...] = y[dayofyear == dayofyear[iday], ...].mean(axis = 0)
        
    return seasonal_cycle
# Note: All of this assumes no nans are present, since they are used inside


def measureDistributional(series, D2 = False):

    if D2:
        axis = 0
    else:
        axis = None

    MEAN = np.array( np.mean(series, axis = axis) )
    P01 = np.array( np.quantile(series, q = (0.01), axis = axis) )
    P99 = np.array( np.quantile(series, q = (0.99), axis = axis) )
    SD = np.array( series.std(axis = axis) )

    return(np.array( (MEAN, SD, P01, P99 ) ))


def validateSeries(series, reference, D2 = False, how = 'relative'):
    # relative:
    # relativePRC:
    # difference:
    # ratio:
    
    series_int = series.copy()
    reference_int = reference.copy()


    if D2:
        axis = 0
    else:
        axis = None

    RMSE = np.array( np.power(np.power((series_int - reference_int), 2).mean(axis = axis), 0.5) )

    series_stats = measureDistributionalPrecip(series_int, D2 = D2)
    reference_stats = measureDistributionalPrecip(reference_int, D2 = D2)
    
    if how == 'relative':
        tbr = (series_stats - reference_stats)/reference_stats
    elif how == 'relativePRC':
        tbr = 100*(series_stats - reference_stats)/reference_stats
    elif how == 'difference':
        tbr = (series_stats - reference_stats)
    elif how == 'ratio':
        tbr = series_stats/reference_stats

    return( (RMSE, tbr) )
        


def measureDistributionalPrecip(series, D2 = False, wet_threshold = 1, stats_over_wet = False):

    if D2:
        axis = 0
    else:
        axis = None
        
    
    series_int = series.copy()

    wet_day_index = (series_int > wet_threshold)
    


    WET_DAYS = np.array( np.nanmean(wet_day_index, axis = axis) )
    
    if stats_over_wet:        
        series_int[ np.logical_not(wet_day_index) ] = np.nan

    MEAN = np.array( np.nanmean(series_int, axis = axis) )
    P01 = np.array( np.nanquantile(series_int, q = (0.01), axis = axis) )
    P05 = np.array( np.nanquantile(series_int, q = (0.05), axis = axis) )
    P25 = np.array( np.nanquantile(series_int, q = (0.25), axis = axis) )
    P50 = np.array( np.nanquantile(series_int, q = (0.5), axis = axis) )
    P75 = np.array( np.nanquantile(series_int, q = (0.75), axis = axis) )
    P95 = np.array( np.nanquantile(series_int, q = (0.95), axis = axis) )
    P99 = np.array( np.nanquantile(series_int, q = (0.99), axis = axis) )
    SD = np.array( np.nanstd(series_int, axis = axis) )
    VAR = np.array( np.nanvar(series_int, axis = axis) )

    return(np.array( (WET_DAYS, MEAN, SD, VAR, P01, P05, P25, P50, P75, P95, P99) ))
    #return( {"WET_DAYS": WET_DAYS, "MEAN": MEAN, "SD": SD, "P01": P01, "P99": P99} )



def validateSeriesPrecip(series, reference, dates = None, D2 = False, 
                         how = 'relative', wet_threshold = 1,
                         stats_over_wet = False):
    
    if D2:
        axis = 0
    else:
        axis = None
        
    series_int = series.copy()
    reference_int = reference.copy()

    RMSE = np.array( np.power( np.mean( np.power((series_int - reference_int), 2), axis = axis ), 0.5) )
    MAE = np.array( np.abs(series_int - reference_int ).mean(axis = axis) )

    #SPCOR = np.zeros(shape = series[0,...].shape, dtype=np.float32)
    #for i in range(SPCOR.shape[0]):
    #    if len(SPCOR.shape) > 1:
    #        for j in range(SPCOR.shape[1]):
    #            SPCOR[i,j] = stats.spearmanr(a = series[:, i, j], b = reference[:, i, j], 
    #                                          nan_policy = 'propagate' ).statistic

    if not (dates is None):
        seasonal_cycle_reference = getSeasonalCycle(reference_int, dates)
        seasonal_cycle_series = getSeasonalCycle(series_int, dates)

        # Plot the seasonal cycle for a random gridpoint
        temporal_correlation = np.zeros(shape = (reference_int.shape[1], reference_int.shape[2]))
        for i in range(reference_int.shape[1]):
            for j in range(reference_int.shape[2]):
                print(i, j)
                temporal_correlation[i,j] = np.corrcoef( (reference_int/seasonal_cycle_reference)[:, i,j],
                                                        (series_int/seasonal_cycle_series)[:, i, j] )[0,1]


    #RMSE_wetreference = np.array( np.power( np.nanmean( np.power((series - reference), 2), axis = axis ), 0.5) )


    # Discretize series_int and reference_int based on the wet_threshold
    series_binary = series_int > wet_threshold
    reference_binary = reference_int > wet_threshold

    # Compute true positives, false positives, true negatives, and false negatives
    TP = np.sum((series_binary == 1) & (reference_binary == 1), axis=axis)
    FP = np.sum((series_binary == 1) & (reference_binary == 0), axis=axis)
    TN = np.sum((series_binary == 0) & (reference_binary == 0), axis=axis)
    FN = np.sum((series_binary == 0) & (reference_binary == 1), axis=axis)

    # Compute rates
    TPR = TP / (TP + FN)  # True Positive Rate
    FPR = FP / (FP + TN)  # False Positive Rate
    TNR = TN / (TN + FP)  # True Negative Rate
    FNR = FN / (FN + TP)  # False Negative Rate


    series_int = measureDistributionalPrecip(series_int, D2 = D2, stats_over_wet = stats_over_wet)
    reference_int = measureDistributionalPrecip(reference_int, D2 = D2, stats_over_wet = stats_over_wet)
    
    if how == 'relative':
        tbr = (series_int - reference_int)/reference_int
    elif how == 'relativePRC':
        tbr = 100*(series_int - reference_int)/reference_int
    elif how == 'difference':
        tbr = (series_int - reference_int)
    elif how == 'ratio':
        tbr = series_int/reference_int
        
        # tbr is (WET_DAYS, MEAN, SD, VAR, P01, P05, P25, P50, P75, P95, P99)

    if not (dates is None):
        print( ["MAE", "RMSE", "TempCOR", "TPR", "FPR", "TNR", "FNR", "WET_DAYS", "MEAN", "SD", "VAR", "P01", "P05", "P25", "P50", "P75", "P95", "P99"])
        return( np.array( [MAE, RMSE, temporal_correlation, TPR, FPR, TNR, FNR,
                          tbr[0], tbr[1], tbr[2], tbr[3], tbr[4], tbr[5], tbr[6], tbr[7], tbr[8], tbr[9], tbr[10]
        ]
                         ) 
               )
    else:
        print( ["MAE", "RMSE", "TPR", "FPR", "TNR", "FNR", "WET_DAYS", "MEAN", "SD", "VAR", "P01", "P05", "P25", "P50", "P75", "P95", "P99"])
        return(  np.array( [MAE, RMSE, TPR, FPR, TNR, FNR, 
                tbr[0], tbr[1], tbr[2], tbr[3], tbr[4], tbr[5], tbr[6], tbr[7], tbr[8], tbr[9], tbr[10]
                ]
               ) )