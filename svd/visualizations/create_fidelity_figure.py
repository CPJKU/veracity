from svd.utils.io import pickle_load
from svd.path_config import project_root, build_localized_explanations_path

from svd.explanations.explanation_utils import THRESHOLD

import pandas as pd
import matplotlib.pyplot as plt

from argparse import ArgumentParser

import seaborn as sns
sns.set_theme(style="whitegrid")

import os


attack_type_title = {
    'spec': 'Spectrogram',
    'wave': 'Waveform'
}

if __name__ == '__main__':
    # This script is used to create Figure 4 of the paper.
    parser = ArgumentParser()
    parser.add_argument("--attack_type", choices=['spec', 'wave'])
    parser.add_argument("--target", type=int, choices=[0, 1])

    args = parser.parse_args()

    attack_type = args.attack_type
    target = args.target

    # these are hardcoded because we only put those in the paper
    # but they are required for loading the right metrics
    subset = 'test'
    k = 1
    baseline = 'mean'

    model_name = 'model_log_0mean'

    metrics = pickle_load(os.path.join(project_root, 'experiments', model_name, 'metrics_localized_{}_{}_{}_{}_k={}.pt'.format(baseline, attack_type, target, subset, k)))
    components = pickle_load(build_localized_explanations_path(model_name, baseline, attack_type, target, subset, k))

    explained_index = 0
    if target == 0:
        # this is because weights are sorted for explaining "sing"
        explained_index = -1

    components_prepped = []
    for comp_row in components:
        explained_weight = [w[0] for w in comp_row[1]][explained_index]
        components_prepped.append(comp_row[0][0] == explained_weight)


    metrics_df = pd.DataFrame(metrics)

    if target == 1:
        metrics_df["success"] = metrics_df["global"] > THRESHOLD
    else:
        metrics_df["success"] = metrics_df["global"] < THRESHOLD

    metrics_df["detected"] = components_prepped

    fig, axes = plt.subplots(1, 1, figsize=(3.2, 3.2))
    axes = [axes]

    df_subset = metrics_df.query('success==True')[["fidelity", "detected"]]
    sns.boxplot(x="detected", y="fidelity", data=df_subset, ax=axes[0])

    axes[0].set_xlabel("Correct Explanation")
    if attack_type == "wave":
        axes[0].set_ylabel("Fidelity")
    else:
        axes[0].set_ylabel("")
    axes[0].set_ylim([0, 1])
    fig.suptitle("")

    fig.tight_layout()
    fig.savefig(os.path.join(project_root, 'figures', 'experiment_fidelity', 'fidelity_{}_{}_{}_{}_k={}.png'.format(baseline, attack_type, target, subset, k)))





