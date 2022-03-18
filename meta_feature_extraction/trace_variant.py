import numpy as np
from scipy import stats
from pm4py.statistics.traces.generic.log import case_statistics


def trace_variant(log):
    variants_count = case_statistics.get_variant_statistics(log)
    variants_count = sorted(variants_count, key=lambda x: x["count"], reverse=True)
    occurrences = [x["count"] for x in variants_count]

    len_occurr, len_log = len(occurrences), len(log)

    ratio_most_common_variant = sum(occurrences[:1]) / len(log)
    ratio_top_1_variants = sum(occurrences[: int(len_occurr * 0.01)]) / len_log
    ratio_top_5_variants = sum(occurrences[: int(len_occurr * 0.05)]) / len_log
    ratio_top_10_variants = sum(occurrences[: int(len_occurr * 0.1)]) / len_log
    ratio_top_20_variants = sum(occurrences[: int(len_occurr * 0.2)]) / len_log
    ratio_top_50_variants = sum(occurrences[: int(len_occurr * 0.5)]) / len_log
    ratio_top_75_variants = sum(occurrences[: int(len_occurr * 0.75)]) / len_log
    mean_variant_occurrence = np.mean(occurrences)
    std_variant_occurrence = np.std(occurrences)
    skewness_variant_occurrence = stats.skew(occurrences)
    kurtosis_variant_occurrence = stats.kurtosis(occurrences)

    return [
        ratio_most_common_variant,
        ratio_top_1_variants,
        ratio_top_5_variants,
        ratio_top_10_variants,
        ratio_top_20_variants,
        ratio_top_50_variants,
        ratio_top_75_variants,
        mean_variant_occurrence,
        std_variant_occurrence,
        skewness_variant_occurrence,
        kurtosis_variant_occurrence,
    ]
