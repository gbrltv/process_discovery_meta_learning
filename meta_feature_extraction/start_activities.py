import numpy as np
from scipy import stats
from pm4py.algo.filtering.log.start_activities import start_activities_filter


def start_activities(log):
    log_start = start_activities_filter.get_start_activities(log)

    n_unique_start_activities = len(log_start)

    start_activities_occurrences = list(log_start.values())
    start_activities_min = np.min(start_activities_occurrences)
    start_activities_max = np.max(start_activities_occurrences)
    start_activities_mean = np.mean(start_activities_occurrences)
    start_activities_median = np.median(start_activities_occurrences)
    start_activities_std = np.std(start_activities_occurrences)
    start_activities_variance = np.var(start_activities_occurrences)
    start_activities_q1 = np.percentile(start_activities_occurrences, 25)
    start_activities_q3 = np.percentile(start_activities_occurrences, 75)
    start_activities_iqr = stats.iqr(start_activities_occurrences)
    start_activities_skewness = stats.skew(start_activities_occurrences)
    start_activities_kurtosis = stats.kurtosis(start_activities_occurrences)

    return [
        n_unique_start_activities,
        start_activities_min,
        start_activities_max,
        start_activities_mean,
        start_activities_median,
        start_activities_std,
        start_activities_variance,
        start_activities_q1,
        start_activities_q3,
        start_activities_iqr,
        start_activities_skewness,
        start_activities_kurtosis,
    ]
