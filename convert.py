import os
import pandas as pd
from tqdm import tqdm
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.conversion.log import converter as log_converter
from meta_feature_extraction import sort_files


path = "event_logs"
path_out = "event_logs_xes"
os.makedirs(path_out, exist_ok=True)

for file in tqdm(sort_files(os.listdir(path))):
    df = pd.read_csv(f"{path}/{file}")
    log = log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG)

    file_name = file.split(".csv")[0]
    xes_exporter.apply(
        log, f"{path_out}/{file_name}.xes", parameters={"show_progress_bar": False}
    )
