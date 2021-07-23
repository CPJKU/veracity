import torch

from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from svd.utils.io import load_model
from svd.core.audio import SVDDataset
from svd.train import prep_net_config
from svd.utils.progress import progress
from svd.core.model import SingingVoiceDetector
from svd.utils.config import parse_config_file, parse_variable_assignments


def opts_parser():
    desc = 'Computes predictions with a neural network trained for singing voice detection.'
    parser = ArgumentParser(description=desc)
    parser.add_argument('modelfile', metavar='MODELFILE', type=str,
                        help='File to load the learned weights from (.npz format)')
    parser.add_argument('outfile', metavar='OUTFILE', type=str,
                        help='File to save the prediction curves to (.npz/.pkl format)')
    parser.add_argument('--dataset', type=str, default='jamendo',
                        help='Name of the dataset to use (default: %(default)s)')
    parser.add_argument('--dataset-path', metavar='DIR', type=str, default=None, required=True,
                        help='Path to data of dataset which will be used')
    parser.add_argument('--pitchshift', metavar='PERCENT', type=float, default=0.0,
                        help='Perform test-time pitch-shifting of given amount and '
                             'direction in percent (e.g., -10 shifts down by 10%%).')
    parser.add_argument('--cache', metavar='DIR', type=str, default=None,
                        help='Store spectra in the given directory (disabled by default)')
    parser.add_argument('--vars', metavar='FILE', action='append', type=str,
                        default=[Path(__file__).parent / 'default.vars'],
                        help='Reads configuration variables from a FILE of KEY=VALUE '
                             'lines. Can be given multiple times, settings from later '
                             'files overriding earlier ones. Will read defaults.vars, then files given here.')
    parser.add_argument('--var', metavar='KEY=VALUE', action='append', type=str,
                        help='Set the configuration variable KEY to VALUE. Overrides ' 
                             'settings from --vars options. Can be given multiple times.')
    parser.add_argument('--loudness', type=float, default=0.0,
                        help='Perform test-time loudness adjustment of given amount and '
                             'direction in decibel (e.g., -3 decreases volume by 3dB).')
    return parser


def get_data_config(config):
    """ Prepares configuration dictionary for validation dataset. """
    data_config = {}
    data_config.update({'fps': config['fps'],
                        'sample_rate': config['sample_rate'],
                        'frame_len': config['frame_len'],
                        'mel_max': config['mel_max']})
    return data_config


def prep_data(dataset_path, dataset, cache, pitch_shift, cfg):
    """ Prepares data loader for validation / test data. """
    data_path = Path(dataset_path)
    info_path = Path(__file__).parent.parent / 'dataset' / dataset
    label_path = info_path / 'labels'
    with open(str(info_path / 'filelists' / 'valid')) as f:
        file_list_val = [data_path / l.rstrip() for l in f if l.rstrip()]
    cached_files_val = [cache / f.with_suffix('.npy').name for f in file_list_val] if cache else None
    with open(str(info_path / 'filelists' / 'test')) as f:
        file_list_test = [data_path / l.rstrip() for l in f if l.rstrip()]
    cached_files_test = [cache / f.with_suffix('.npy').name for f in file_list_test] if cache else None

    # prepare data loaders
    valid_data = SVDDataset(file_list_val, cached_files_val, label_path, get_data_config(cfg), pitch_shift)
    valid_data = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=8)
    if all([f.exists() for f in file_list_test]):
        test_data = SVDDataset(file_list_test, cached_files_test, label_path, get_data_config(cfg), pitch_shift)
        test_data = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)
    else:
        test_data = None
    return valid_data, test_data


def do_prediction(data, net, device, config, loudness=0.0):
    """ Gets predictions for given data. """
    net.eval()
    block_len = config['blocklen']
    padding = torch.zeros((block_len // 2, config['frame_len'] // 2 + 1), dtype=torch.float)[None, None, ...]
    predictions = []

    for x, y in progress(data, desc='Predicting file '):
        # - adjust loudness if needed
        if loudness:
            x = x * float(10. ** (loudness / 10.))
        x = torch.cat((padding, x, padding), dim=-2)
        x = x.to(device)

        # pass full spectrogram through network
        pred = net.probabilities(x)
        predictions.append(pred)

    return predictions


def run_prediction(valid_data, test_data, config, model_path, spec_len, output_file, loudness):
    """ Prepares network, obtains predictions for validation / test data and stores predictions. """
    print('Preparing network for prediction...')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net_config = prep_net_config(config, spec_len)
    net = load_model(model_path, SingingVoiceDetector(net_config), device)

    valid_predictions = do_prediction(valid_data, net, device, config, loudness)
    if test_data:
        test_predictions = do_prediction(test_data, net, device, config, loudness)
        files = valid_data.dataset.files + test_data.dataset.files
        predictions = valid_predictions + test_predictions
    else:
        files = valid_data.dataset.files
        predictions = valid_predictions

    # save predictions
    data = {str(f): p.cpu().numpy() for f, p in zip(files, predictions)}
    print('Saving predictions...')
    if output_file.endswith('.pkl'):
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(data, f, protocol=-1)
    else:
        import numpy as np
        np.savez(output_file, **data)


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    modelfile = options.modelfile
    outfile = options.outfile

    # read configuration files and immediate settings
    cfg = {}
    if Path(modelfile + '.vars').exists():
        options.vars.insert(1, modelfile + '.vars')
    for fn in options.vars:
        cfg.update(parse_config_file(fn))
    cfg.update(parse_variable_assignments(options.var))

    # prepare cache
    cache = Path(options.cache) if options.cache else None
    if cache and not cache.exists():
        cache.mkdir(parents=True)

    # prepare paths and files
    valid_data, test_data = prep_data(options.dataset_path, options.dataset, cache, options.pitchshift, cfg)

    bin_nyquist = cfg['frame_len'] // 2 + 1
    bin_mel_max = bin_nyquist * 2 * cfg['mel_max'] // cfg['sample_rate']

    run_prediction(valid_data, test_data, cfg, modelfile, bin_mel_max, outfile, options.loudness)


if __name__ == '__main__':
    main()
