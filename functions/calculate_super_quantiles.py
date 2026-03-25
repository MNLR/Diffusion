import numpy as np

def calculate_super_quantiles(data):
    q5 = np.nanpercentile(data, 5)
    q95 = np.nanpercentile(data, 95)
    super_q5 = np.nanmean(data[data <= q5])
    super_q95 = np.nanmean(data[data >= q95])
    return super_q5, super_q95
