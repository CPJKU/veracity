from svd.explanations.explanation_utils import setup, create_composition_predict_fn_explanation, composition_fn_specLIME, prepare_factorization_input_spec
from svd.explanations.explanation_utils import compute_explanation

from svd.path_config import horses_path, build_model_path
from svd.utils.io import pickle_load, pickle_dump
import os
import warnings

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--baseline", default="min", choices=["min", "zero", "mean"])
    parser.add_argument("--which_horses", required=True, choices=['horses1', 'horses2', 'horses3'])
    args = parser.parse_args()

    first_layer = "0mean"

    baseline = args.baseline
    which_horses = args.which_horses

    model_name = 'model_log_{}'.format(first_layer)

    model_file = build_model_path(model_name)
    model, spec_transform, cfg = setup(model_file, magscale='log', firstconv_zeromean=first_layer)

    _, predict_fn_specLIME = create_composition_predict_fn_explanation(model, take_first_element=True)

    hop_size = cfg['sample_rate'] // cfg['fps']
    sample_rate = cfg['sample_rate']

    horses_specs = pickle_load(os.path.join(horses_path, which_horses+'.pt'))
    explanations = []

    for i, (orig, horse) in enumerate(horses_specs):
        orig = prepare_factorization_input_spec(orig, model, to_tensor=True)
        horse = prepare_factorization_input_spec(horse, model, to_tensor=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explanations_orig = compute_explanation(orig, composition_fn_specLIME, predict_fn_specLIME,
                                                    sample_rate, hop_size, baseline=baseline, targets=[0, 1])
            explanations_adv = compute_explanation(horse, composition_fn_specLIME, predict_fn_specLIME,
                                                   sample_rate, hop_size, baseline=baseline, targets=[0, 1])
        explanations.append([(explanations_orig, explanations_adv)])
    explanations_path = os.path.join(horses_path, 'explanations_{}_{}.pt'.format(which_horses, baseline))
    pickle_dump(explanations, explanations_path)
