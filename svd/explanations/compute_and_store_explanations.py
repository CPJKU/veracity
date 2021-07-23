import warnings
from svd.explanations.explanation_utils import compute_explanation, setup, create_composition_predict_fn_explanation, \
    device, prepare_factorization_input_spec

from argparse import ArgumentParser

from svd.path_config import build_adversary_path, build_explanation_path, build_model_path
from svd.utils.io import pickle_dump, pickle_load

if __name__ == '__main__':
    # compute explanations for a list of adversaries
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

    model_name = 'model_log_{}'.format(first_layer)
    model_file = build_model_path(model_name)
    model, _, cfg = setup(model_file, magscale='log', firstconv_zeromean=first_layer)

    hop_size = cfg['sample_rate'] // cfg['fps']
    sample_rate = cfg['sample_rate']

    _, predict_fn_specLIME = create_composition_predict_fn_explanation(model, take_first_element=True)

    composition_fn_specLIME = lambda x: x.transpose(0, 1).unsqueeze(0).to(device)

    adversaries_path = build_adversary_path(model_name, attack_type, target, subset)


    adversaries = pickle_load(adversaries_path)
    explanations = []
    for i in range(len(adversaries)):
        adversary_explanations = []
        for j in range(len(adversaries[i])):
            original_spec, adversarial_spec = adversaries[i][j]
            original_spec = prepare_factorization_input_spec(original_spec, model)
            adversarial_spec = prepare_factorization_input_spec(adversarial_spec, model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                expls_orig = compute_explanation(original_spec, composition_fn_specLIME, predict_fn_specLIME,
                                                 sample_rate, hop_size, baseline=baseline, targets=[0, 1])
                expls_adv = compute_explanation(adversarial_spec, composition_fn_specLIME, predict_fn_specLIME,
                                                sample_rate, hop_size, baseline=baseline, targets=[0, 1])
            adversary_explanations.append((expls_orig, expls_adv))
        explanations.append(adversary_explanations)
    pickle_dump(explanations,
                build_explanation_path(model_name, baseline, attack_type, target, subset))
