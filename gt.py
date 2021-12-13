import numpy as np
import scipy.stats

def gt(a, alpha=0.05, rep=0):
    b = a
    b[np.isinf(a)] = np.nan

    outlier = 1

    while outlier:
        tmp = b[~np.isnan(b)]
        meanval = np.mean(tmp)
        meantmp = np.abs(tmp - meanval)
        maxval = tmp[np.where(meantmp == max(meantmp))]
        maxval = maxval[0]
        sdval = np.std(tmp)
        tn = np.abs((maxval - meanval) / sdval)

        size = len(tmp)
        t_dist = scipy.stats.t.ppf(1 - alpha / (2 * size), size - 2)
        numerator = (size - 1) * np.sqrt(np.square(t_dist))
        denominator = np.sqrt(size) * np.sqrt(size - 2 + np.square(t_dist))
        critval = numerator / denominator

        outlier = (tn > critval)

        if outlier:
            tmp = np.where(a == maxval)
            b[tmp] = np.nan

    idx = np.where(np.isnan(b))
    outliers = a[idx]

    if ~rep:
        b = b[~np.isnan(b)]

    return b, idx, outliers
