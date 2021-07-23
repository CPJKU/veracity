import os
import torch
from svd.explanations.explanation_utils import device, preprocess_adversaries, setup, npyfy
from svd.path_config import build_model_path, horses_path, project_root
from svd.utils.io import pickle_load
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import librosa.display

font = {'size' : 14}
plt.rc('font', **font)

if __name__ == '__main__':
    # This script is used to create Figure 1 of the paper.
    parser = ArgumentParser()
    parser.add_argument("--which_horses", required=True, choices=['horses1', 'horses2', 'horses3'])
    parser.add_argument("--horse_index", required=True, type=int)
    args = parser.parse_args()

    which_horses = args.which_horses
    horse_index = args.horse_index

    baseline = "mean" # fixed

    # load horses
    horses_specs = pickle_load(os.path.join(horses_path, "{}.pt".format(which_horses)))
    explanations_path = os.path.join(horses_path, 'explanations_{}_{}.pt'.format(which_horses, baseline))
    explanations = pickle_load(explanations_path)

    # load model
    first_layer = "0mean"
    model_name = 'model_log_{}'.format(first_layer)
    model_file = build_model_path(model_name)
    model, spec_transform, cfg = setup(model_file, magscale='log', firstconv_zeromean=first_layer)

    def prepare_visualization_input_spec(x):
        x = torch.tensor(x)
        x = x.to(device)
        x = model.pre_proc(x)
        x = model.mag_trans(x)
        x = x.squeeze().transpose(0, 1)
        x = npyfy(x)
        return x


    adversaries = []
    for tup in horses_specs:
        adversaries.append([tup])

    processed_adversaries = preprocess_adversaries(adversaries, prepare_visualization_input_spec)

    results_orig, results_adv = explanations[horse_index][0]
    _, metrics_orig, _ = results_orig
    expls_adv, metrics_adv, _ = results_adv
    orig, horse = processed_adversaries[horse_index][0]

    print(metrics_orig['global'], metrics_adv['global'])

    print("min:", min(orig.min(), horse.min()))
    print("max:", max(orig.max(), horse.max()))

    params = {
        'vmin': -8.1517, # min of the 3 selected
        'vmax': 4.414 # max of the 3 selected
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 2))
    librosa.display.specshow(orig, ax=axes[0], cmap='magma', **params)
    axes[0].set_title("f(original)={:.2f}".format(metrics_orig['global']))
    librosa.display.specshow(horse, ax=axes[1], cmap='magma', **params)
    axes[1].set_title("f(adversarial)={:.2f}".format(metrics_adv['global']))

    img = librosa.display.specshow(npyfy(expls_adv[1]), ax=axes[2], cmap='magma', **params)
    axes[2].set_title("fidelity={:.2f}".format(metrics_adv['fidelity']))

    # source: https://stackoverflow.com/a/13784887/1117932
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(img, cax=cbar_ax)

    print(metrics_adv)
    fig.savefig(os.path.join(project_root, 'figures', 'experiment_horses', "{}_idx{}_{}.png".format(which_horses, str(horse_index).zfill(2), baseline)),
                bbox_inches='tight')
