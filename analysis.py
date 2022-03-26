import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mode
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from pymfe.mfe import MFE
import warnings


def heatmap(df, metrics, outpath="analysis"):
    """
    Heatmap of average positions for each quality criteria
    """

    final_rank = df_models.groupby("variant")[metrics].mean()
    final_rank.columns = [
        "Fitness",
        "Precision",
        "Generalization",
        "Simplicity",
        "Time",
    ]
    final_rank.index.name = "Discovery Algorithms"

    plt.figure(figsize=(5, 2.5))
    img = sns.heatmap(final_rank, annot=True, cmap="YlGnBu", annot_kws={"fontsize": 12})
    plt.yticks(rotation=0)
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(f"{outpath}/5_algorithms_rank_5metrics.pdf", dpi=300)


def heatmap_2(df, metrics, outpath="analysis"):
    """
    Heatmap of average positions for each quality criteria (different coloring)
    """

    final_rank = df_models.groupby("variant")[metrics].mean()
    final_rank.columns = ["Fitness", "Precision", "Generalization", "Simplicity"]
    final_rank.index.name = "Discovery Algorithm"

    plt.figure(figsize=(4.5, 2.25))
    img = sns.heatmap(
        final_rank, annot=True, cmap=sns.cm.rocket_r, annot_kws={"fontsize": 12}
    )
    plt.yticks(rotation=0)
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(f"{outpath}/3_algorithms_rank_4metrics.pdf", dpi=300)


def create_meta_database(df_targets, df_features, metrics):
    """
    Given the meta-features and possible targets, the function ranks the performances and associate the best algorithm to an instance.
    This is repeated for all event logs. The product is the meta-database.
    """
    # Rotulating meta-target using the mean rankings
    df_models["target_final_rank"] = df_models.filter(metrics).mean(axis=1)

    # Finding the best techniques (ranking = min)
    best = pd.DataFrame(df_models.groupby("log")["target_final_rank"].min())
    best.reset_index(level=0, inplace=True)

    # Filtering meta-instances where target_final_rank = min
    meta_target = df_models.set_index(["log", "target_final_rank"]).join(
        best.set_index("log", "target_final_rank")
    )
    meta_target.reset_index(level=0, inplace=True)
    meta_target = meta_target[meta_target.index == meta_target.target_final_rank]
    meta_target.reset_index(drop=True, inplace=True)

    # Removing repetitions
    meta_target.drop_duplicates(subset="log", keep=False, inplace=True)

    meta_database = (
        meta_target.filter(["log", "variant"])
        .set_index(["log"])
        .join(df_logs.set_index("log"))
    )
    meta_database.reset_index(level=0, inplace=True)

    print(f"{bold}Creating meta-database{end}")
    print(
        "#instances:",
        len(meta_database),
        "| #classes (discovery):",
        len(meta_database["variant"].unique()),
    )
    print("Number of meta-target appearances")
    print(Counter(meta_database["variant"]), end="\n\n")

    return meta_database


def meta_learning_exp1(X, y, outpath="analysis"):
    """
    The meta-learning approach predicts the best discovery technique based on log features.
    This function implements a 30-fold cross-validation strategy combined with a holdout of 75%/25%.
    The meta-model is inferred using the training data and its performance is retrieved using the test data.
    This experiment contains 5 classes (meta-targets) and algorithms are ranked according to 5 quality criteria.
    """
    out = []
    for step in range(30):
        # splitting data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=step
        )

        # meta-model
        rf = RandomForestClassifier(random_state=step, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        out.append(
            [
                accuracy_score(y_test, y_pred),
                f1_score(y_test, y_pred, average="weighted"),
                "Meta-Model",
            ]
        )

        # random approach
        random_y = np.random.randint(len(y.unique()), size=(1, len(y_test)))[0]
        out.append(
            [
                accuracy_score(y_test, random_y),
                f1_score(y_test, random_y, average="weighted"),
                "Random",
            ]
        )

        # majority approach (AM)
        majority_y = [mode(y_test)] * len(y_test)
        out.append(
            [
                accuracy_score(y_test, majority_y),
                f1_score(y_test, majority_y, average="weighted"),
                "Majority",
            ]
        )

    result_df = pd.DataFrame(out, columns=["Acc", "F1", "Metric"])
    plot_df = pd.melt(result_df, id_vars="Metric")
    plot_df.columns = ["Method", "Metric", "Performance"]

    result_df = pd.DataFrame(out, columns=["Acc", "F1", "Metric"])
    for method in plot_df["Method"].unique():
        for metric in plot_df["Metric"].unique():
            perfs = list(
                plot_df.loc[
                    (plot_df["Method"] == method) & (plot_df["Metric"] == metric),
                    "Performance",
                ]
            )
            print(
                f"{method} {metric}: {np.round(np.mean(perfs), 2)} ({np.round(np.std(perfs), 2)})"
            )
    print()

    plt.figure(figsize=(5, 2))
    ax = sns.boxplot(
        x="Performance", y="Method", hue="Metric", data=plot_df, palette="YlGnBu"
    )
    ax.xaxis.grid(True)
    plt.tight_layout()
    plt.savefig(f"{outpath}/5_algorithms_boxplot_5metrics.pdf", dpi=300)
    plt.close()


def loo_cv(df):
    """
    Leave-one-out cross validation.
    For each instance IN, trains a model with all instances except IN and predicts IN's class.
    """
    out = []
    for i in range(len(df)):
        test = df.loc[i].copy()
        train = df.drop(i, axis=0).copy()
        X_train, y_train = train.drop(["log", "variant"], axis=1), train["variant"]
        X_test, y_test = test.drop(["log", "variant"]), test["variant"]

        rf = RandomForestClassifier(random_state=10, n_jobs=-1)
        rf.fit(X_train.to_numpy(), y_train)
        y_pred = rf.predict(X_test.to_numpy().reshape(1, -1))[0]
        out.append([test["log"], y_test, y_pred])

    return pd.DataFrame(out, columns=["log", "label", "pred"])


def plot_confusion_matrix(y_true, y_pred, labels, outpath="analysis"):
    """
    Plotting confusion matrix for LOOCV experiment
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.tight_layout()
    plt.savefig(f"{outpath}/leave_one_out_confusion_matrix.pdf", dpi=300)
    plt.close()


def pca_analysis(meta_database, loo_cv, outpath="analysis"):
    """
    Dimensionality reduction analysis with PCA. We also plot correct/incorrect classification results.
    """
    meta_database.insert(1, "classification", "")
    meta_database.iloc[
        meta_database[loo_cv["label"] == loo_cv["pred"]].index, 1
    ] = "Correct"
    meta_database.iloc[
        meta_database[loo_cv["label"] != loo_cv["pred"]].index, 1
    ] = "Incorrect"

    pca = PCA(n_components=2)
    X_new = pca.fit_transform(
        Normalizer().fit_transform(
            meta_database.drop(["log", "classification", "variant"], axis=1).to_numpy()
        )
    )
    print(
        f"Explained variance ratio: PC 1 ({np.round(pca.explained_variance_ratio_[0]*100, 2)}%) and PC 2 ({np.round(pca.explained_variance_ratio_[1]*100, 2)}%)"
    )
    print("Plotting PCA", end="\n\n")
    df_pca = pd.DataFrame(X_new, columns=["PC1", "PC2"])
    df_pca.insert(2, "Classification", meta_database["classification"])
    df_pca.insert(2, "Meta-target", meta_database["variant"])

    sns.set_theme()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        ax=ax,
        data=df_pca,
        x="PC1",
        y="PC2",
        hue="Meta-target",
        style="Classification",
        palette="bright",
        alpha=0.9,
        hue_order=["AM", "HM", "IM", "IMf", "IMd"],
        markers={"Correct": "o", "Incorrect": "X"},
        style_order=["Correct", "Incorrect"],
        s=100,
    )
    ax.set_xlabel(
        f"PC1 ({np.round(pca.explained_variance_ratio_[0]*100, 2)}% explained variance)"
    )
    ax.set_ylabel(
        f"PC2 ({np.round(pca.explained_variance_ratio_[1]*100, 2)}% explained variance)"
    )
    plt.tight_layout()
    plt.savefig(f"{outpath}/pca.pdf", dpi=300)
    plt.close()


def complexity_analysis(df):
    """
    Metrics to measure classification complexity
    """
    warnings.simplefilter("ignore")
    mfe = MFE(groups="complexity", features=["f1v", "n2"])
    mfe.fit(
        X=df.drop(["log", "variant"], axis=1).to_numpy(),
        y=list(df["variant"]),
        transform_num=False,
    )
    ft = mfe.extract(suppress_warnings=True, divide="ignore")
    complexity = dict(zip(ft[0], ft[1]))

    print("F1v:", complexity["f1v.mean"])
    print("N2:", complexity["n2.mean"], end="\n\n")


def meta_learning_exp2(X, y, outpath="analysis"):
    """
    The meta-learning approach predicts the best discovery technique based on log features.
    This function implements a 30-fold cross-validation strategy combined with a holdout of 75%/25%.
    The meta-model is inferred using the training data and its performance is retrieved using the test data.
    This experiment contains 3 classes (meta-targets) and algorithms are ranked according to 4 quality criteria.
    """
    out = []
    for step in range(30):
        # splitting data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=step
        )

        # meta-model
        rf = RandomForestClassifier(random_state=step, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        out.append(
            [
                accuracy_score(y_test, y_pred),
                f1_score(y_test, y_pred, average="weighted"),
                "Meta-Model",
            ]
        )

        # random approach
        random_y = np.random.randint(len(y.unique()), size=(1, len(y_test)))[0]
        out.append(
            [
                accuracy_score(y_test, random_y),
                f1_score(y_test, random_y, average="weighted"),
                "Random",
            ]
        )

        # majority approach (AM)
        majority_y = [mode(y_test)] * len(y_test)
        out.append(
            [
                accuracy_score(y_test, majority_y),
                f1_score(y_test, majority_y, average="weighted"),
                "Majority",
            ]
        )

    result_df = pd.DataFrame(out, columns=["Acc", "F1", "Metric"])
    plot_df = pd.melt(result_df, id_vars="Metric")
    plot_df.columns = ["Method", "Metric", "Performance"]

    result_df = pd.DataFrame(out, columns=["Acc", "F1", "Metric"])
    for method in plot_df["Method"].unique():
        for metric in plot_df["Metric"].unique():
            perfs = list(
                plot_df.loc[
                    (plot_df["Method"] == method) & (plot_df["Metric"] == metric),
                    "Performance",
                ]
            )
            print(
                f"{method} {metric}: {np.round(np.mean(perfs), 2)} ({np.round(np.std(perfs), 2)})"
            )
    print()

    plt.figure(figsize=(5, 2))
    ax = sns.boxplot(
        x="Performance", y="Method", hue="Metric", data=plot_df, palette="autumn_r"
    )
    ax.xaxis.grid(True)
    plt.tight_layout()
    plt.savefig(f"{outpath}/3_algorithms_boxplot_4metrics.pdf", dpi=300)
    plt.close()


def plot_feature_importance(X, y, exp="exp1", outpath="analysis"):
    """
    Analyzing and plotting the feature importances.
    """
    rf = RandomForestClassifier(random_state=43)
    rf.fit(X, y)

    importances = pd.DataFrame(rf.feature_importances_)
    importances = pd.concat([pd.DataFrame(X.columns), importances], axis=1)
    importances.columns = ["Feature", "Importance"]

    importances_top = importances.sort_values("Importance", ascending=False)
    importances_bottom = importances.sort_values("Importance", ascending=True)

    plt.figure(figsize=(4, 2.5))
    ax = sns.barplot(
        x="Importance", y="Feature", data=importances_top[:10], color="g", orient="h"
    )
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("", fontsize=1)
    ax.set_xticks([0, 0.02, 0.04])
    ax.xaxis.grid(True)
    plt.tight_layout()
    if exp == "exp1":
        plt.savefig(f"{outpath}/5_algorithms_Importances_TOP_5metrics.pdf")
    elif exp == "exp2":
        plt.savefig(f"{outpath}/3_algorithms_Importances_TOP_4metrics.pdf")

    plt.figure(figsize=(4, 2.5))
    ax = sns.barplot(
        x="Importance", y="Feature", data=importances_bottom[:10], color="r", orient="h"
    )
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("", fontsize=1)
    ax.set_xticks([0, 0.02, 0.04])
    ax.xaxis.grid(True)
    plt.tight_layout()
    if exp == "exp1":
        plt.savefig(f"{outpath}/5_algorithms_Importances_Bottom_5metrics.pdf")
    elif exp == "exp2":
        plt.savefig(f"{outpath}/3_algorithms_Importances_Bottom_4metrics.pdf")


def sample_cut(data, cut_type, frac, n_samples, seed=1, print_bins=True, n_bins=10):
    """
    This function bins the instances based on the given heuristics: cut_type, frac and n_samples
    """
    if cut_type == "cut":
        data["entropy_bins"] = pd.cut(
            data["activities_q1"], bins=n_bins, labels=range(1, n_bins + 1)
        )
    elif cut_type == "qcut":
        data["entropy_bins"] = pd.qcut(
            data["activities_q1"], q=n_bins, labels=range(1, n_bins + 1)
        )
    else:
        print("Cut type not defined")
        return None
    frames = []
    for bin in range(1, n_bins + 1):
        binned_df = data[data["entropy_bins"] == bin]
        if n_samples and len(binned_df) < n_samples:
            selected_samples = binned_df
        else:
            selected_samples = binned_df.sample(
                frac=frac, n=n_samples, random_state=seed
            )

        frames.append(selected_samples)
        if print_bins:
            print(
                f"bin {bin} - original size: {len(binned_df)} - sampled size: {len(selected_samples)}"
            )

    sampled_data = pd.concat(frames, ignore_index=True)

    del sampled_data["entropy_bins"]
    del data["entropy_bins"]

    return sampled_data


def classify(data, print_perf=False):
    X = data.drop(["log", "variant"], axis=1)
    y = data["variant"].astype("category").cat.codes

    out = []
    for step in range(30):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=step)

        # meta-model
        rf = RandomForestClassifier(random_state=step, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        out.append(
            [
                accuracy_score(y_test, y_pred),
                f1_score(y_test, y_pred, average="weighted"),
                "Meta-Model",
            ]
        )

        # random approach
        random_y = np.random.randint(len(y.unique()), size=(1, len(y_test)))[0]
        out.append(
            [
                accuracy_score(y_test, random_y),
                f1_score(y_test, random_y, average="weighted"),
                "Random",
            ]
        )

        # majority approach (AM)
        majority_y = [Counter(y_test).most_common()[0][0]] * len(y_test)
        out.append(
            [
                accuracy_score(y_test, majority_y),
                f1_score(y_test, majority_y, average="weighted"),
                "Majority",
            ]
        )

    result_df = pd.DataFrame(out, columns=["Acc", "F1", "Metric"])
    plot_df = pd.melt(result_df, id_vars="Metric")
    plot_df.columns = ["Method", "Metric", "Performance"]
    list_perf = []
    for method in plot_df["Method"].unique():
        for metric in plot_df["Metric"].unique():
            perfs = list(
                plot_df.loc[
                    (plot_df["Method"] == method) & (plot_df["Metric"] == metric),
                    "Performance",
                ]
            )
            avg, std = np.round(np.mean(perfs), 2), np.round(np.std(perfs), 2)
            list_perf.extend([avg, std])
            if print_perf:
                print(f"{method} {metric}: {avg} ({std})")

    return list_perf


def resampling_exp(df):
    performances = []
    for n_samples in [5, 10, 20, 30, 40, 50]:
        out = []
        for seed in range(10):
            sampled_meta_database = sample_cut(
                df,
                cut_type="cut",
                frac=None,
                n_samples=n_samples,
                seed=seed,
                print_bins=False,
            )
            inner = [n_samples]
            inner.extend(classify(sampled_meta_database))
            out.append(inner)
        performances.append(list(np.round(np.mean(np.array(out), axis=0), 2)))

    cols = [
        "sample",
        "mm_acc",
        "mm_acc_std",
        "mm_f1",
        "mm_f1_std",
        "rand_acc",
        "rand_acc_std",
        "rand_f1",
        "rand_f1_std",
        "maj_acc",
        "maj_acc_std",
        "maj_f1",
        "maj_f1_std",
    ]
    return pd.DataFrame(performances, columns=cols)


# Create output directory
out_path = "analysis"
os.makedirs(out_path, exist_ok=True)
bold = "\033[1m"
end = "\033[0m"

print(f"{bold}Loading data{end}")
# Importing meta-features
df_logs = pd.read_csv("log_meta_features.csv")

# Importing discovery metrics
df_models = pd.read_csv("discovery_metrics.csv")

print(f"{bold}Ranking algorithms{end}")
# Creating ranks for each metric (fitness, precision, generalization, simplicity, time)
df_models["discovery_time_rank"] = df_models.groupby("log")["discovery_time"].rank(
    method="min", ascending=True, na_option="bottom"
)
df_models["fitness_rank"] = df_models.groupby("log")["log_fitness"].rank(
    method="min", ascending=False, na_option="bottom"
)
df_models["precision_rank"] = df_models.groupby("log")["precision"].rank(
    method="min", ascending=False, na_option="bottom"
)
df_models["generalization_rank"] = df_models.groupby("log")["generalization"].rank(
    method="min", ascending=False, na_option="bottom"
)
df_models["simplicity_rank"] = df_models.groupby("log")["simplicity"].rank(
    method="min", ascending=False, na_option="bottom"
)

metrics = [
    "fitness_rank",
    "precision_rank",
    "generalization_rank",
    "simplicity_rank",
    "discovery_time_rank",
]

# Plotting positional heatmaps
heatmap(df_models, metrics)

# Meta-database creation
df_meta_database = create_meta_database(df_models, df_logs, metrics)

# Preparing meta-database
X = df_meta_database.drop(["log", "variant"], axis=1)
y = df_meta_database["variant"].astype("category").cat.codes

print(f"{bold}Meta-learning prediction{end}", end="\n\n")
meta_learning_exp1(X, y)

print(f"{bold}Analyzing feature importance{end}", end="\n\n")
plot_feature_importance(X, y)

print(f"{bold}Leave-one-out experiment{end}")
df_loo_cv = loo_cv(df_meta_database)

# performances
y_true, y_pred = df_loo_cv["label"], df_loo_cv["pred"]
print("Meta-Model Acc:", np.round(accuracy_score(y_true, y_pred), 2))
print("Meta-Model F1:", np.round(f1_score(y_true, y_pred, average="weighted"), 2))

# Confusion matrix
labels = ["AM", "HM", "IM", "IMd", "IMf"]
plot_confusion_matrix(y_true, y_pred, labels)

# Classification report
print("Performance per meta-target")
print(classification_report(y_true, y_pred, target_names=labels), end="\n\n")

print(f"{bold}Dimensionality analysis (PCA with 2 components){end}")
pca_analysis(df_meta_database.copy(), df_loo_cv)

print(f"{bold}Complexity analysis{end}")
complexity_analysis(df_meta_database)

print(f"{bold}Pipeline using only traditional discovery algorithms{end}")
print("Dropping IMf and IMd", end="\n\n")

# Removing IMf and IMd
df_models = df_models[df_models["variant"] != "IMf"].copy()
df_models = df_models[df_models["variant"] != "IMd"].copy()

print(f"{bold}Ranking algorithms (excluding discovery time){end}")
# Creating ranks for each metric (fitness, precision, generalization, simplicity)
df_models["fitness_rank"] = df_models.groupby("log")["log_fitness"].rank(
    method="min", ascending=False, na_option="bottom"
)
df_models["precision_rank"] = df_models.groupby("log")["precision"].rank(
    method="min", ascending=False, na_option="bottom"
)
df_models["generalization_rank"] = df_models.groupby("log")["generalization"].rank(
    method="min", ascending=False, na_option="bottom"
)
df_models["simplicity_rank"] = df_models.groupby("log")["simplicity"].rank(
    method="min", ascending=False, na_option="bottom"
)

metrics = [
    "fitness_rank",
    "precision_rank",
    "generalization_rank",
    "simplicity_rank",
]

heatmap_2(df_models, metrics)

df_meta_database_exp2 = create_meta_database(df_models, df_logs, metrics)

X = df_meta_database_exp2.drop(["log", "variant"], axis=1)
y = df_meta_database_exp2["variant"].astype("category").cat.codes

print(f"{bold}Meta-learning prediction (Exp. 2){end}", end="\n\n")
meta_learning_exp2(X, y)
print(f"{bold}Analyzing feature importance{end}", end="\n\n")
plot_feature_importance(X, y, "exp2")

print(f"{bold}Evaluating internal validity of experiments{end}")
print(f"{bold}Resampling strategy{end}")
df_samp_results = resampling_exp(df_meta_database)
df_samp_results = df_samp_results.loc[:, ["sample", "mm_acc", "mm_f1"]].copy()
df_samp_results.columns = ["#Samples per bin", "Accuracy", "F-score"]
print(df_samp_results)
