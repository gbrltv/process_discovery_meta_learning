import os
import time
import pandas as pd
from tqdm import tqdm
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.conformance.tokenreplay.variants import token_replay
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from meta_feature_extraction import sort_files


columns = [
    "log",
    "variant",
    "discovery_time",
    "fitness_time",
    "precision_time",
    "generalization_time",
    "simplicity_time",
    "fitness",
    "precision",
    "generalization",
    "simplicity",
]

algorithms = ["AM", "HM", "IM", "IMf", "IMd"]


def discover_model(algorithm):
    """
    Apply the corresponding discovery algorithm and returns a tuple containing the petri net, the initial marking and the final marking.
    """
    if algorithm == "AM":
        return alpha_miner.apply(
            log, variant=alpha_miner.Variants.ALPHA_VERSION_CLASSIC
        )
    elif algorithm == "HM":
        return heuristics_miner.apply(log)
    elif algorithm == "IM":
        return inductive_miner.apply(log, variant=inductive_miner.Variants.IM)
    elif algorithm == "IMf":
        return inductive_miner.apply(log, variant=inductive_miner.Variants.IMf)
    elif algorithm == "IMd":
        return inductive_miner.apply(log, variant=inductive_miner.Variants.IMd)
    return None


def extract_metrics(log, algorithm):
    """
    Given a log and a discovery algorithm, returns model quality metrics.
    """
    start_time = time.time()
    net, im, fm = discover_model(algorithm)
    discovery_time = time.time() - start_time

    start_time = time.time()
    fitness = replay_fitness_evaluator.apply(
        log,
        net,
        im,
        fm,
        variant=replay_fitness_evaluator.Variants.TOKEN_BASED,
        parameters={"show_progress_bar": False},
    )
    fitness_time = time.time() - start_time

    start_time = time.time()
    precision = precision_evaluator.apply(
        log,
        net,
        im,
        fm,
        variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN,
        parameters={"show_progress_bar": False},
    )
    precision_time = time.time() - start_time

    start_time = time.time()
    generalization = generalization_evaluator.apply(
        log, net, im, fm, parameters={"show_progress_bar": False}
    )
    generalization_time = time.time() - start_time

    start_time = time.time()
    simplicity = simplicity_evaluator.apply(net)
    simplicity_time = time.time() - start_time

    return [
        discovery_time,
        fitness_time,
        precision_time,
        generalization_time,
        simplicity_time,
        fitness["log_fitness"],
        precision,
        generalization,
        simplicity,
    ]


model_metrics = []
event_logs_path = "event_logs_xes"
for file in tqdm(sort_files(os.listdir(event_logs_path))):
    log_name = file.split(".xes")[0]
    log = xes_importer.apply(
        f"{event_logs_path}/{file}", parameters={"show_progress_bar": False}
    )

    for algo in algorithms:
        model_metrics.append([log_name, algo, *extract_metrics(log, algo)])

pd.DataFrame(model_metrics, columns=columns).to_csv(
    "discovery_metrics.csv", index=False
)
