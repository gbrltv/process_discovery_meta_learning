from pm4py.algo.filtering.log.variants import variants_filter


def simple_stats(log):
    n_traces = len(log)

    variants = variants_filter.get_variants(log)
    n_unique_traces = len(variants)

    ratio_unique_traces_per_trace = n_unique_traces / n_traces

    return [n_traces, n_unique_traces, ratio_unique_traces_per_trace]
