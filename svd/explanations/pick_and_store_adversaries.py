import numpy as np
from svd.explanations.explanation_utils import setup, THRESHOLD as threshold
from svd.path_config import cache_path, dataset_path, build_adversary_path, build_model_path, \
    build_adversarial_examples_path
from svd.predict import prep_data
from svd.utils.io import pickle_dump
from svd.adversaries.log_analysis import get_successful_snippets, read_file, _get_data_config, str_to_bool
from pathlib import Path
import torch
import os
from argparse import ArgumentParser
from svd.adversaries.adv_audio import get_feature


def pick_adversaries_from_successful_snippets(model, healthy_data, adv_path, threshold,
                                              target_idx, n_adversaries=10, adversary_extension=".wav",
                                              block_len=115):

    padding_offset = block_len // 2

    def predict(x):
        return model.probabilities(x)

    data_config = _get_data_config()

    np.random.seed(2021)
    adversaries = []
    for i, (x, label) in enumerate(iter(healthy_data)):

        adv_sample_path = os.path.join(adv_path, os.path.basename(healthy_data.dataset.files[i]).replace(".ogg", adversary_extension).replace(".mp3", adversary_extension))
        successful_snippets = list(get_successful_snippets(Path(adv_sample_path)))

        print("found:", len(successful_snippets))

        if adversary_extension == ".npy":
            adv_sample = torch.tensor(np.load(adv_sample_path))
        else:
            data = read_file(adv_sample_path, data_config['sample_rate'])
            adv_sample = get_feature(data.unsqueeze(0), data_config)

        i_adversaries = []

        count_i = 0
        while len(i_adversaries) < n_adversaries and count_i < 100000:
            count_i += 1
            random_idx = np.random.randint(0, len(successful_snippets)) # pick one random successful snippet
            x_stored, gt_label, orig_pred, adv_pred_, block_start = successful_snippets[random_idx]
            x_stored = x_stored.cuda()

            if str_to_bool(adv_pred_) != target_idx:  # skipping adversaries in the wrong direction
                continue
            x_snip = x[:, :, block_start - padding_offset:block_start - padding_offset + block_len, :].cuda()

            adversarial_snippet = adv_sample[..., block_start:block_start+block_len, :].cuda()

            print(adversarial_snippet.shape, x_stored.shape)
            diff = adversarial_snippet - x_stored
            print("diff", diff.min(), diff.max())

            try:

                print(predict(adversarial_snippet), predict(x_snip), predict(x_stored))
                adv_pred = predict(adversarial_snippet) > threshold
                orig_pred = predict(x_snip) > threshold
                if adv_pred == target_idx and orig_pred != target_idx:
                    print("found an adversary at {}:{} {}->{}".format(i, block_start, orig_pred, adv_pred))
                    i_adversaries.append((x_snip.squeeze(),
                                          adversarial_snippet.squeeze()))
                else:
                    print("skipping {}/{}/{}".format(adv_pred, orig_pred, adv_pred_))

            except:
                print('caught exception')
                pass
        adversaries.append(i_adversaries)
    return adversaries



if __name__ == '__main__':
    # This script is used to sanity check the computation of adversarial examples, and to randomly
    # pick examples for the experiments.
    parser = ArgumentParser()
    parser.add_argument("--attack_type", required=True, choices=["wave", "spec"])
    parser.add_argument("--subset", required=True, choices=["valid", "test"])
    parser.add_argument("--target", required=True, type=int, choices=[0, 1])
    args = parser.parse_args()

    first_layer = "0mean"
    target = args.target
    attack_type = args.attack_type
    subset = args.subset

    type_extension_mapping = {
        'spec': '.npy',
        'wave': '.wav'
    }

    adv_path = build_adversarial_examples_path(attack_type, first_layer, subset)
    extension = type_extension_mapping[attack_type]
    print(adv_path, extension)

    model_name = 'model_log_{}'.format(first_layer)
    model_file = build_model_path(model_name)
    model, spec_transform, cfg = setup(model_file, magscale='log', firstconv_zeromean=first_layer)

    valid_data, test_data = prep_data(dataset_path, "jamendo", Path(cache_path), 0.0, cfg)

    original_data = valid_data
    if subset == "test":
        original_data = test_data

    adversaries = pick_adversaries_from_successful_snippets(model, original_data, adv_path, threshold, target,
                                                            adversary_extension=extension)
    pickle_dump(adversaries, build_adversary_path(model_name, attack_type, target, subset))
