import os
import pandas as pd
from tqdm import tqdm
from meta_feature_extraction import pipeline, sort_files


with open("meta_feature_extraction/column_names.txt") as f:
    columns = f.readlines()
columns = [x.strip() for x in columns]

path = "event_logs_xes"
combined_features = []

print("Extracting meta-features")
for file in tqdm(sort_files(os.listdir(path))):
    combined_features.append(pipeline(path, file))

pd.DataFrame(combined_features, columns=columns).to_csv(
    "log_meta_features.csv", index=False
)
