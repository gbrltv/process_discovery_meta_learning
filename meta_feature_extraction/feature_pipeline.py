from pm4py.objects.log.importer.xes import importer as xes_importer
from .simple_stats import simple_stats
from .trace_length import trace_length
from .trace_variant import trace_variant
from .activities import activities
from .start_activities import start_activities
from .end_activities import end_activities
from .entropies import entropies


def pipeline(event_logs_path, log_name):
    log = xes_importer.apply(
        f"{event_logs_path}/{log_name}", parameters={"show_progress_bar": False}
    )

    features = [log_name.split(".xes")[0]]
    features.extend(simple_stats(log))
    features.extend(trace_length(log))
    features.extend(trace_variant(log))
    features.extend(activities(log))
    features.extend(start_activities(log))
    features.extend(end_activities(log))
    features.extend(entropies(log_name, event_logs_path))

    return features
