from svd.path_config import build_adversary_path, build_explanation_path, build_model_path, \
    build_localized_explanations_path
from svd.utils.io import pickle_load, pickle_dump
from svd.explanations.explanation_utils import TEMPORAL_SEGMENTS, composition_fn_specLIME, \
    create_composition_predict_fn_explanation, setup, preprocess_adversaries, \
    compute_all_modifications, compute_explanation, prepare_factorization_input_spec

import torch
from audioLIME.factorization_slime import TimeFrequencyTorchFactorization

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--attack_type", required=True, choices=["wave", "spec"])
    parser.add_argument("--subset", required=True, choices=["valid", "test"])
    parser.add_argument("--baseline", default="min", choices=["min", "zero", "mean"])
    parser.add_argument("--target", required=True, type=int, choices=[0, 1])
    args = parser.parse_args()

    first_layer = "0mean"

    attack_type = args.attack_type
    subset = args.subset
    baseline = args.baseline
    target = args.target

    kw_param = 'kw025'
    mag_scale = "log"

    model_name = 'model_log_{}'.format(first_layer)

    # load model
    model_file = build_model_path(model_name)
    model, spec_transform, cfg = setup(model_file, magscale='log', firstconv_zeromean=first_layer)

    hop_size = cfg['sample_rate'] // cfg['fps']

    _, predict_fn_specLIME = create_composition_predict_fn_explanation(model)

    def predict_fn_specLIME_nparray(x):
        return predict_fn_specLIME(x[0])

    def compose_and_predict(x):
        return predict_fn_specLIME(composition_fn_specLIME(x))

    # load precomputed adversaries and explanations
    adversaries = pickle_load(build_adversary_path(model_name, attack_type, target, subset))
    explanations = pickle_load(build_explanation_path(model_name, baseline, attack_type, target, subset))

    binary_factorization = TimeFrequencyTorchFactorization(
        torch.ones_like(prepare_factorization_input_spec(adversaries[1][0][0], model)),
        frequency_segments=4, temporal_segmentation_params=TEMPORAL_SEGMENTS,
        hop_length=hop_size,
        target_sr=cfg['sample_rate'], composition_fn=composition_fn_specLIME, baseline="zero")

    def get_components_collected(adversaries, k_components):
        components_collected = []
        predictions = []
        metrics = []
        for sample_idx in range(len(adversaries)):
            print("processing sample {}/{}".format(sample_idx + 1, len(adversaries)))
            for explanation_idx in range(len(adversaries[sample_idx])):
                original_spec, adversarial_spec = adversaries[sample_idx][explanation_idx]
                original_pred, adversarial_pred, pred_slime, pred_obv, mod_delta_slime, mod_delta_obv, top_weights, \
                largest_indeces = \
                    compute_all_modifications(original_spec, adversarial_spec, compose_and_predict,
                                              explanations[sample_idx][explanation_idx],
                                              binary_factorization, k_components=k_components, target_idx=target,
                                              compute_norm_on='delta', verbose=False)

                explanation_mod_delta_obv = compute_explanation(original_spec + mod_delta_obv, composition_fn_specLIME,
                                                                predict_fn_specLIME_nparray, sample_rate=cfg['sample_rate'],
                                                                hop_size=hop_size,
                                                                baseline=baseline, targets=[0, 1], sort_by_index=False)
                predictions.append(pred_obv[1])  # P(sing)
                print(explanation_mod_delta_obv[1])
                metrics.append(explanation_mod_delta_obv[1])
                explanation_obv = explanation_mod_delta_obv[2]
                components_collected.append((largest_indeces.tolist(), explanation_obv))
        return components_collected, predictions, metrics

    processed_adversaries_factorization = preprocess_adversaries(adversaries, prepare_factorization_input_spec, model=model)

    for k in [1, 3, 5]:
        print("processing k={}".format(k))
        components_collected, predictions, metrics = get_components_collected(processed_adversaries_factorization, k)
        storage_path = build_localized_explanations_path(model_name, baseline, attack_type,
                                                         target, subset, k)
        pickle_dump(components_collected, storage_path)

        pickle_dump(predictions,
                    storage_path.replace("components_localized_", "predictions_localized_"))
        pickle_dump(metrics,
                    storage_path.replace("components_localized_", "metrics_localized_"))
