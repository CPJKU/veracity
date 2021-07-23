from svd.path_config import build_localized_explanations_path, build_figure_perturbation_path
from svd.utils.io import pickle_load
from svd.explanations.explanation_utils import THRESHOLD
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
import pandas as pd
import numpy as np

from argparse import ArgumentParser

font = {'size': 22}
plt.rc('font', **font)

if __name__ == '__main__':
    # This script is used to create Figure 3 of the paper.
    parser = ArgumentParser()
    parser.add_argument("--attack_type", required=True, choices=["wave", "spec"])
    parser.add_argument("--subset", required=True, choices=["valid", "test"])
    parser.add_argument("--baseline", default="min", choices=["min", "zero", "mean"])
    parser.add_argument("--target", required=True, type=int, choices=[0, 1])
    parser.add_argument("--cumulative", action='store_true', default=False)
    args = parser.parse_args()

    first_layer = "0mean"

    attack_type = args.attack_type
    subset = args.subset
    baseline = args.baseline
    target = args.target
    cumulative = args.cumulative

    model_name = 'model_log_{}'.format(first_layer)

    predictions = {
        1: [], 3: [], 5: []
    }

    components_collected_k = {}
    for k in [1, 3, 5]:
        storage_path = build_localized_explanations_path(model_name, baseline, attack_type, target, subset, k)
        components_collected_k[k] = pickle_load(storage_path)
        predictions[k] = pickle_load(storage_path.replace("components_localized_", "predictions_localized_"))

    print(components_collected_k[5][0])
    print(components_collected_k[3][0])

    components_aggregated = {
        1: [], 3: [], 5: []
    }

    for k in [1, 3, 5]:
        components_current_k = components_collected_k[k]
        for comp_row in components_current_k:
            explanation_indeces = comp_row[1]
            if target == 0:
                explanation_indeces = explanation_indeces[::-1]
            explained_weights = [w[0] for w in explanation_indeces[:k]]
            components_aggregated[k].append(len(set(comp_row[0]).intersection(explained_weights)))

    df_ = pd.concat([pd.DataFrame({"components": components_aggregated[1], "predictions": predictions[1], "k": 1}),
                     pd.DataFrame({"components": components_aggregated[3], "predictions": predictions[3], "k": 3}),
                     pd.DataFrame({"components": components_aggregated[5], "predictions": predictions[5], "k": 5})])

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5), sharey=True)

    for i, k in enumerate([1, 3, 5]):
        query_comparator = '>'
        if target == 0:
            query_comparator = '<'
        df_subset = df_.query("predictions{}{} and k=={}".format(query_comparator, THRESHOLD, k))
        print(df_subset.head())
        print(attack_type, target, k, len(df_subset))
        if cumulative:
            sns.ecdfplot(data=df_subset, x="components", stat="proportion", ax=axes[i])
        else:
            sns.countplot(x="components", data=df_subset, ax=axes[i])
        axes[i].set_xlim((-0.5, k+0.5))
        axes[i].invert_xaxis()
        axes[i].set_xticks(np.arange(k + 1))
        axes[i].set_xticklabels(np.arange(k+1))
        axes[i].set_ylabel("Count")
        axes[i].set_xlabel("Detected Segments")
        axes[i].set_title("Modified Segments: {}".format(k))
    fig.tight_layout()

    figure_path = build_figure_perturbation_path(baseline, attack_type, target, subset, cumulative)
    fig.savefig(figure_path)
