import numpy as np
from scipy import stats
from pm4py.algo.filtering.log.end_activities import end_activities_filter


def end_activities(log):
    log_end = end_activities_filter.get_end_activities(log)

    n_unique_end_activities = len(log_end)

    end_activities_occurrences = list(log_end.values())
    end_activities_min = np.min(end_activities_occurrences)
    end_activities_max = np.max(end_activities_occurrences)
    end_activities_mean = np.mean(end_activities_occurrences)
    end_activities_median = np.median(end_activities_occurrences)
    end_activities_std = np.std(end_activities_occurrences)
    end_activities_variance = np.var(end_activities_occurrences)
    end_activities_q1 = np.percentile(end_activities_occurrences, 25)
    end_activities_q3 = np.percentile(end_activities_occurrences, 75)
    end_activities_iqr = stats.iqr(end_activities_occurrences)
    end_activities_skewness = stats.skew(end_activities_occurrences)
    end_activities_kurtosis = stats.kurtosis(end_activities_occurrences)

    return [
        n_unique_end_activities,
        end_activities_min,
        end_activities_max,
        end_activities_mean,
        end_activities_median,
        end_activities_std,
        end_activities_variance,
        end_activities_q1,
        end_activities_q3,
        end_activities_iqr,
        end_activities_skewness,
        end_activities_kurtosis,
    ]
