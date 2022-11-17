# Meta-learning for Process Discovery

> This file lists the steps to reproduce the experiments, analysis and figures generated for the paper "[Automating Process Discovery Through Meta-learning](https://link.springer.com/chapter/10.1007/978-3-031-17834-4_12)" published at the CoopIS 2022 conference, ensuring reproducibility.

This work proposes a meta-learning approach to map event log profiles, discovery algorithms, and quality criteria. We propose a set of 93 meta-features extracted from the event logs used as meta-instances in our experiment. The meta-features capture process behavior at different levels, such as activity, trace, and log. Using meta-learning, we show that it is possible to leverage the process model quality by automatically recommending appropriate discovery algorithms considering the process behavior.

## Contents

This repository already comes with the datasets employed in the experiments along with the code to reproduce them. We also provide the experimental results (see *.csv* files) and figures used in the papers (see *analysis* folder).


## Installation steps

First, you need to install conda to manage the environment. See installation instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

The next step is to create the environment. For that, run:

```shell
conda create --name bpm_experiments python=3.9.0
```

Then, activate the environment:

```shell
conda activate bpm_experiments
```

Finally, install the dependencies:

```shell
python -m pip install -r requirements.txt
```


## Reproducing experiments

The first step is to convert the event logs to the *xes* format using the following command:

```shell
python convert.py
```

The above command generates the *event_logs_xes* folder containing the converted event logs. With that, we proceed to the next step:

```shell
python extract_features.py
```

This extracts the event log meta-features and store them in the *log_meta_features.csv* file. Next, we apply the process discovery algorithms for all event logs and extract performance metrics to measure the quality of discovered models:

```shell
python discovery.py
```

As a result, the *discovery_metrics.csv* file will be generated. Finally, we move to:

```shell
python analysis.py
```

This command generates the main experiments used in the paper. The products are (i) the *analysis* folder containing the figures used in the paper and (ii) the terminal outputs listing performances and experiment details.
