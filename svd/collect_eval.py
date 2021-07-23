import os
import numpy as np
import torch
import pandas as pd

from pathlib import Path
from argparse import ArgumentParser
from svd.eval import evaluate, load_labels
from svd.path_config import dataset_path

def opts_parser():
    parser = ArgumentParser(description='Evaluates singing voice predictions against ground truth.')
    parser.add_argument('--experiment_root', metavar='EXPROOT', type=str,
                        help='Directory to load the prediction curves from (.npz/.pkl format).',
                        default='/share/cp/projects/singing_voice_detection/experiments/')
    parser.add_argument('--dataset', type=str, default='jamendo',
                        help='Name of the dataset to use (default: %(default)s)')
    parser.add_argument('--smooth-width', metavar='WIDTH', type=int, default=56,
                        help='Apply temporal smoothing over WIDTH frames (default: (default)s)')
    return parser


def get_prediction_files(experiment_path):
    all_files = os.listdir(experiment_path)
    prediction_files = [f for f in all_files if f.startswith("predictions") and f.endswith(".npz")]
    return prediction_files


def read_prediction_file(model_path, pf):
    preds = np.load(os.path.join(model_path, pf))
    preds = {Path(f): torch.tensor(preds[f]) for f in preds.keys()}
    return preds


def create_printable_results(results, model_name, pf, threshold):
    printable_results = {'model_name': model_name,
                         'loudness': pf.replace("predictions_", "").replace(".npz", ""),
                         'threshold': threshold.item()
                         }
    for key, value in results.items():
        printable_results[key] = value.mean().item() * 100
    printable_results['error'] = 100 - printable_results['accuracy']
    return printable_results

def main():
    parser = opts_parser()
    options = parser.parse_args()
    fps = 70

    experiment_root = options.experiment_root
    models = os.listdir((experiment_root))
    print("found the following experiments:", models)

    # general paths
    data_path = Path(dataset_path)
    info_path = Path(__file__).parent.parent / 'dataset' / options.dataset
    label_path = info_path / 'labels'
    with open(str(info_path / 'filelists' / 'valid')) as f:
        file_list_val = [data_path / l.rstrip() for l in f if l.rstrip()]
    with open(str(info_path / 'filelists' / 'test')) as f:
        file_list_test = [data_path / l.rstrip() for l in f if l.rstrip()]

    # iterate overall found models
    collected_results = []
    for model_name in models:
        model_path = os.path.join(experiment_root, model_name)
        prediction_files = get_prediction_files(model_path)

        # compute threshold on normal validation set
        preds = read_prediction_file(model_path, 'predictions_normal.npz')
        threshold, _ = evaluate([preds[fn].squeeze() for fn in file_list_val],
                 load_labels(file_list_val, preds, fps, label_path),
                 smoothen=options.smooth_width)

        print("model: ", model_name)
        print('| loudness | threshold | Precision | Recall | Specificity | F1 | error | AUROC |')
        for pf in prediction_files:
            preds = read_prediction_file(model_path, pf)

            threshold, results = evaluate([preds[fn].squeeze() for fn in file_list_test],
                                          load_labels(file_list_test, preds, fps, label_path),
                                          smoothen=options.smooth_width, thres=threshold, comp_auroc=True)
            printable_results = create_printable_results(results, model_name, pf, threshold)
            print('| %s | %.2f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | ' % (
                printable_results['loudness'], printable_results['threshold'],
                printable_results['precision'], printable_results['recall'],
                printable_results['specificity'], printable_results['fscore'],
                printable_results['error'], printable_results['auroc']
            ))
            collected_results.append(printable_results)

    df_results = pd.DataFrame(collected_results)
    df_results.to_csv(os.path.join(experiment_root, "collected_results.csv"))

if __name__ == '__main__':
    main()