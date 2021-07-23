from svd.explanations.explanation_utils import setup, create_composition_predict_fn_explanation, preprocess_adversaries, THRESHOLD, evaluate_component_impact, composition_fn_specLIME, TEMPORAL_SEGMENTS, \
    prepare_factorization_input_spec
from svd.path_config import build_adversary_path, build_explanation_path, build_model_path
from svd.utils.io import pickle_load

from audioLIME.factorization_slime import TimeFrequencyTorchFactorization

import torch

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--attack_type", choices=['spec', 'wave'], required=True)
    parser.add_argument("--baseline", choices=['min', 'mean', 'zero'], required=True)
    parser.add_argument("--target", type=int, choices=[0, 1], required=True)

    args = parser.parse_args()

    attack_type = args.attack_type
    baseline = args.baseline
    target = args.target

    # fixed:
    subset = "test"
    kw_param = 'kw025'
    first_layer = "0mean"
    magscale = "log"

    model_name = 'model_log_{}'.format(first_layer)
    model_file = build_model_path(model_name)
    model, spec_transform, cfg = setup(model_file, magscale='log', firstconv_zeromean=first_layer)

    _, predict_fn_specLIME = create_composition_predict_fn_explanation(model)
    compose_and_predict = lambda x: predict_fn_specLIME(composition_fn_specLIME(x))

    hop_size = cfg['sample_rate'] // cfg['fps']

    # load adversaries
    adversaries = pickle_load(build_adversary_path(model_name, attack_type, target, subset))
    explanations = pickle_load(build_explanation_path(model_name, baseline, attack_type, target, subset))

    # here we use the factorization object as a binary mask
    # by setting baseline "zero" and the content to torch.ones we get a binary mask with 1's
    # at the desired segments
    binary_factorization = TimeFrequencyTorchFactorization(
        torch.ones_like(prepare_factorization_input_spec(adversaries[0][0][0], model)),
        frequency_segments=4, temporal_segmentation_params=TEMPORAL_SEGMENTS,
        hop_length=hop_size,
        target_sr=cfg['sample_rate'], composition_fn=composition_fn_specLIME, baseline="zero")

    preprocess_adversaries_factorization = preprocess_adversaries(adversaries, prepare_factorization_input_spec, model=model)
    flipped_slime, flipped_norm_based = evaluate_component_impact(preprocess_adversaries_factorization, explanations,
                                                                  compose_and_predict,
                                                                  binary_factorization, THRESHOLD,
                                                                  k_components='positive', compute_norm_on="delta",
                                                                  target_idx=target)

    n_explanations = sum([len(sub_list) for sub_list in explanations])
    print("attack_type: {} target: {}, slime: {:.2f} norm: {:.2f}".format(attack_type, target,
                                                                          flipped_slime.sum() / n_explanations,
                                                                          flipped_norm_based.sum() / n_explanations))