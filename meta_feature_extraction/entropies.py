import os
import subprocess


def default_call(log_name, arg1, arg2, arg3, path):
    output = subprocess.run(
        [
            "java",
            "-jar",
            f"{os.getcwd()}/meta_feature_extraction/eventropy.jar",
            arg1,
            arg2,
            arg3,
            f"{os.getcwd()}/{path}/{log_name}",
        ],
        capture_output=True,
        text=True,
    )
    if len(output.stdout) == 0:
        return 0
    return float(output.stdout.strip().split(":")[1])


def entropies(log_name, path):
    single_args = ["-f", "-p", "-B", "-z"]
    double_args = ["-d", "-r"]

    entrops = []
    for arg in single_args:
        entrops.append(default_call(log_name, arg, "", "", path))
    for arg in double_args:
        for i in ["1", "3", "5"]:
            entrops.append(default_call(log_name, arg, i, "", path))
    for i in ["3", "5", "7"]:
        entrops.append(default_call(log_name, "-k", i, "1", path))

    return entrops
