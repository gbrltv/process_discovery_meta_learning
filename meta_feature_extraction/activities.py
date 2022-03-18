import numpy as np
from scipy import stats
from pm4py.algo.filtering.log.attributes import attributes_filter


def activities(log):
    activities = attributes_filter.get_attribute_values(log, "concept:name")
    n_unique_activities = len(activities)

    activities_occurrences = list(activities.values())
    activities_min = np.min(activities_occurrences)
    activities_max = np.max(activities_occurrences)
    activities_mean = np.mean(activities_occurrences)
    activities_median = np.median(activities_occurrences)
    activities_std = np.std(activities_occurrences)
    activities_variance = np.var(activities_occurrences)
    activities_q1 = np.percentile(activities_occurrences, 25)
    activities_q3 = np.percentile(activities_occurrences, 75)
    activities_iqr = stats.iqr(activities_occurrences)
    activities_skewness = stats.skew(activities_occurrences)
    activities_kurtosis = stats.kurtosis(activities_occurrences)

    return [
        n_unique_activities,
        activities_min,
        activities_max,
        activities_mean,
        activities_median,
        activities_std,
        activities_variance,
        activities_q1,
        activities_q3,
        activities_iqr,
        activities_skewness,
        activities_kurtosis,
    ]
