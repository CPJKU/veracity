from svd.path_config import build_figure_flipcount_path, build_model_path, \
    build_adversary_path, build_explanation_path
from svd.explanations.explanation_utils import setup, create_composition_predict_fn_explanation, composition_fn_specLIME, \
    TEMPORAL_SEGMENTS, preprocess_adversaries, evaluate_component_impact, THRESHOLD
from svd.utils.io import pickle_load
import torch
from audioLIME.factorization_slime import TimeFrequencyTorchFactorization
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("whitegrid")

font = {'size' : 22}
plt.rc('font', **font)

attack_type_title = {
    'spec': 'Spectrogram',
    'wave': 'Waveform'
}
attack_target = ['no sing', 'sing']

from matplotlib.ticker import MaxNLocator

from argparse import ArgumentParser

if __name__ == '__main__':
    # This script is used to create Figure 2 of the paper.
    parser = ArgumentParser()
    parser.add_argument("--attack_type", required=True, choices=["wave", "spec"])
    parser.add_argument("--subset", required=True, choices=["valid", "test"])
    parser.add_argument("--baseline", default="min", choices=["min", "zero", "mean"])
    parser.add_argument("--target", required=True, type=int, choices=[0, 1])
    parser.add_argument("--plot_legend", action="store_true", default=False)
    args = parser.parse_args()

    first_layer = "0mean"

    attack_type = args.attack_type
    subset = args.subset
    baseline = args.baseline
    target = args.target
    plot_legend = args.plot_legend

    model_name = 'model_log_{}'.format(first_layer)

    model_file = build_model_path(model_name)
    model, spec_transform, cfg = setup(model_file, magscale='log', firstconv_zeromean=first_layer)

    _, predict_fn_specLIME = create_composition_predict_fn_explanation(model)

    def predict_fn_specLIME_nparray(x):
        return predict_fn_specLIME(x[0])

    def prepare_factorization_input_spec(x):
        x = x.unsqueeze(0)
        x = model.pre_proc(x)
        x = model.mag_trans(x)
        x = x.squeeze().transpose(0, 1)
        return x

    hop_size = cfg['sample_rate'] // cfg['fps']
    compose_and_predict = lambda x: predict_fn_specLIME(composition_fn_specLIME(x))

    adversaries = pickle_load(build_adversary_path(model_name, attack_type, target, subset))
    explanations = pickle_load(build_explanation_path(model_name, baseline, attack_type, target, subset))

    binary_factorization = TimeFrequencyTorchFactorization(
        torch.ones_like(prepare_factorization_input_spec(adversaries[1][0][0])),
        frequency_segments=4, temporal_segmentation_params=TEMPORAL_SEGMENTS,
        hop_length=hop_size,
        target_sr=cfg['sample_rate'], composition_fn=composition_fn_specLIME, baseline="zero")

    processed_adversaries_factorization = preprocess_adversaries(adversaries, prepare_factorization_input_spec)
    compute_norm_on = 'delta'

    counts_slime = []
    counts_norm_based = []
    for k in range(1, 21):
        flipped_slime, flipped_norm_based = evaluate_component_impact(processed_adversaries_factorization, explanations,
                                                                      compose_and_predict,
                                                                      binary_factorization, THRESHOLD, k_components=k,
                                                                      compute_norm_on=compute_norm_on, target_idx=target)
        counts_slime.append(flipped_slime.mean()*100)
        counts_norm_based.append(flipped_norm_based.mean()*100)
        if k == 3:
            print("{} {} lime: {}".format(attack_type, target, flipped_slime.mean()*100))

    ax = plt.figure(figsize=(6.4, 4)).gca()

    plt.plot(range(1, 21), counts_slime, '--^', markersize=12, label="LIME")
    plt.plot(range(1, 21), counts_norm_based, '--o', markersize=12, label="norm-based")
    # plot_title = '{}_{}_bl:{}_t:{}_norm:{}'.format(attack_type, subset, baseline, target, compute_norm_on)
    plot_title = '{} | target: {}'.format(attack_type_title[attack_type], attack_target[target])
    plt.ylabel('% labels flipped')

    plt.yticks([0, 20, 40, 60, 80, 100])

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel('# segments')
    # plt.title(plot_title)
    if plot_legend:
        plt.legend(title="Component selection")
    plt.tight_layout()
    plt.savefig(build_figure_flipcount_path(baseline, attack_type, target, subset))