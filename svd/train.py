import torch

from torch import nn
from torch import optim
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from svd.utils.progress import progress_data
from svd.core.model import SingingVoiceDetector
from svd.core.audio import IterSVDDataset, SVDDataset
from svd.utils.io import save_model, save_errors, save_config
from svd.utils.config import parse_config_file, parse_variable_assignments


def opts_parser():
    """ Prepares argument parsing. """
    desc = 'Trains a NN for singing voice detection with Pytorch, based on https://github.com/f0k/ismir2015.'
    parser = ArgumentParser(description=desc)
    parser.add_argument('modelfile', metavar='MODELFILE', type=str,
                        help='File to save the learned weights to (.pt format)')
    parser.add_argument('--dataset', type=str, default='jamendo',
                        help='Name of the dataset to use (default: %(default)s)')
    parser.add_argument('--dataset-path', metavar='DIR', type=str, default=None, required=True,
                        help='Path to data of dataset which will be used')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Perform train-time data augmentation (enabled by default)')
    parser.add_argument('--no-augment', action='store_false', dest='augment',
                        help='Disable train-time data augmentation')
    parser.add_argument('--validate', action='store_true', default=False,
                        help='Monitor validation loss (disabled by default)')
    parser.add_argument('--save-errors', action='store_true', default=False,
                        help='If given, save error log in {MODELFILE%%.npz}.err.npz.')
    parser.add_argument('--cache', metavar='DIR', type=str, default=None,
                        help='Cache files in the given directory (disabled by default)')
    parser.add_argument('--vars', metavar='FILE', action='append', type=str,
                        default=[Path(__file__).parent / 'default.vars'],
                        help='Reads configuration variables from a FILE of KEY=VALUE '
                             'lines. Can be given multiple times, settings from later '
                             'files overriding earlier ones. Will read defaults.vars, '
                             'then files given here.')
    parser.add_argument('--var', metavar='KEY=VALUE', action='append', type=str,
                        help='Set the configuration variable KEY to VALUE. Overrides '
                             'settings from --vars options. Can be given multiple times.')
    return parser


def prep_data_config(config, augment):
    """ Prepares dictionary containing arguments for data loading. """
    data_config = {}
    data_config.update({'augment': augment})
    # arguments for loading / computing spectrogram
    data_config.update({'sample_rate': config['sample_rate'],
                        'fps': config['fps'],
                        'frame_len': config['frame_len']})
    # arguments for extracting random excerpts
    frames = int(config['blocklen'] / (1 - config['max_stretch'])) if augment else config['blocklen']
    bin_nyquist = config['frame_len'] // 2 + 1
    if config['filterbank'] == 'mel_learn':
        bin_mel_max = bin_nyquist
    else:
        bin_mel_max = bin_nyquist * 2 * config['mel_max'] // config['sample_rate']
    data_config.update({'batch_size': config['batchsize'],
                        'frames': frames,
                        'bins': None if augment else bin_mel_max})
    # arguments for stretch / shift
    data_config.update({'max_stretch': config['max_stretch'],
                        'max_shift': config['max_shift'],
                        'keep_frames': config['blocklen'],
                        'keep_bins': int(config['frame_len'] // 2 + 1),
                        'order': config['spline_order']})
    # arguments for random frequency filters
    data_config.update({'max_freq': config['mel_max'],
                        'max_db': config['max_db']})
    return data_config, bin_mel_max


def prep_net_config(config, spec_len):
    """ Prepares dictionary containing arguments for SVD-network. """
    net_config = {}
    # parameters for Mel transform
    net_config.update({'n_mels': config['mel_bands'],
                       'sample_rate': config['sample_rate'],
                       'f_min': config['mel_min'],
                       'f_max': config['mel_max'],
                       'n_stft': int(config['frame_len'] // 2 + 1),
                       'frame_len': config['frame_len'],
                       'filterbank': config['filterbank'],
                       'spec_len': spec_len})

    # parameters for magnitude scaling
    net_config.update({'magscale': config['magscale'],
                       'arch.batch_norm': config['arch.batch_norm'],
                       'arch.convdrop': config['arch.convdrop'],
                       'arch': config['arch']})

    # arch params
    net_config.update({
        'arch.firstconv_zeromean': config.get('arch.firstconv_zeromean', 'std')
    })

    return net_config


def get_optimiser(params, config):
    """ Prepares optimiser based on configurations. """
    lr = config['initial_eta']
    l2_decay = config.get('l2_decay', 0)
    if config['learn_scheme'] == 'nesterov':
        return optim.SGD(params, lr=lr, momentum=config['momentum'], nesterov=True, weight_decay=l2_decay)
    elif config['learn_scheme'] == 'momentum':
        return optim.SGD(params, lr=lr, momentum=config['momentum'], nesterov=False, weight_decay=l2_decay)
    elif config['learn_scheme'] == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=l2_decay)
    else:
        raise ValueError('Invalid learn-scheme defined!')


def do_train_epoch(desc, epoch_size, net, train_data, criterion, optimiser, device, save_loss):
    """ Performs one epoch of training. """
    net.train()
    t_loss = 0.

    for x, y in progress_data(train_data, desc=desc, total=epoch_size, min_delay=.5):
        x, y = x.to(device), y.to(device)
        y = (0.02 + 0.96 * y)  # map 0 -> 0.02, 1 -> 0.98
        logits = net(x)

        loss = criterion(logits, y)
        t_loss += loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    tot_loss = t_loss / epoch_size
    print('Train loss: %.3f' % tot_loss)
    return tot_loss if save_loss else None


def do_valid_epoch(net, valid_data, criterion, device, save_loss, config):
    """ Performs one epoch of validation. """
    from svd.eval import evaluate
    net.eval()
    predictions = []
    v_loss = 0.

    max_len = config['fps'] * 30

    for x, y in valid_data:
        x, y = x.to(device), y.to(device)
        # choose central 30 seconds
        excerpt = slice(max(0, (x.shape[1] - max_len) // 2), (x.shape[1] + max_len) // 2)
        x = x[None, :, excerpt, :]
        y = y[excerpt][config['blocklen'] // 2:-(config['blocklen'] // 2)].float()
        logits = net(x)

        loss = criterion(logits, y)
        v_loss += loss.item()
        predictions.append((net.probabilities_from_logits(logits), y))

    tot_loss = v_loss / len(valid_data)
    print('Validation loss: %.3f' % tot_loss)
    _, results = evaluate(*zip(*predictions))
    print('Validation error:%.3f' % (1 - results['accuracy']))
    return (tot_loss, results) if save_loss else (None, None)


def run_training(train_data, valid_data, cfg, spec_len, model_save_path, save):
    """ Prepares network, runs training / validation loop and saves trained network. """
    print('Preparing network...')
    net_config = prep_net_config(cfg, spec_len)
    net = SingingVoiceDetector(net_config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimiser = get_optimiser(net.parameters(), cfg)
    lr_scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=cfg.get('eta_decay_every', 1), gamma=cfg['eta_decay'])
    epoch_size = cfg['epochsize']

    train_losses, valid_losses, valid_accs = [], [], []

    # run main training / validation loop
    for ep in range(1, cfg['epochs'] + 1):
        desc = 'Epoch %d/%d: Batch ' % (ep, cfg['epochs'])
        train_loss = do_train_epoch(desc, epoch_size, net, train_data, criterion, optimiser, device, save)
        train_losses.append(train_loss)
        if valid_data:
            valid_loss, valid_acc = do_valid_epoch(net, valid_data, criterion, device, save, cfg)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)

        lr_scheduler.step()

    # save final model
    print('Finished learning; saving model...')
    save_model(model_save_path, net)
    # also save errors (if necessary)
    if save:
        # save train / valid losses, validation accuracy
        save_errors(model_save_path.parent / (model_save_path.stem + '_train_loss.err.npz'), train_losses)
        if valid_data:
            save_errors(model_save_path.parent / (model_save_path.stem + '_valid_loss.err.npz'), valid_losses)
            save_errors(model_save_path.parent / (model_save_path.stem + '_valid_acc.err.npz'), valid_accs)


def main():
    print('Starting configuration...')
    # parse arguments
    parser = opts_parser()
    options = parser.parse_args()
    modelfile = Path(options.modelfile)

    # check if parent of modelfile exists, create otherwise
    if not modelfile.parent.exists():
        modelfile.parent.mkdir(parents=True)

    # read configuration files and immediate settings
    cfg = {}
    for fn in options.vars:
        cfg.update(parse_config_file(fn))
    cfg.update(parse_variable_assignments(options.var))
    # save parameter setting
    save_config(modelfile.with_suffix('.vars'), cfg)
    # get data configuration
    data_config, spec_len = prep_data_config(cfg, options.augment)

    # prepare paths and files
    data_path = Path(options.dataset_path)
    info_path = Path(__file__).parent.parent / 'dataset' / options.dataset
    lab_file = info_path / 'labels'
    cache = Path(options.cache) if options.cache else None
    if cache and not cache.exists():
        cache.mkdir(parents=True)

    # load file list and prepare data
    print('Preparing data loader...')
    with open(str(info_path / 'filelists' / 'train')) as f:
        file_list = [data_path / l.rstrip() for l in f if l.rstrip()]
    cached_files = [cache / (f.name.rsplit('.')[0] + '.npy') for f in file_list] if cache else None
    train_data = IterSVDDataset(file_list, cached_files, lab_file, data_config)
    train_data = DataLoader(train_data, batch_size=None, shuffle=False, num_workers=8)

    if options.validate:
        # prepare validation data
        with open(str(info_path / 'filelists' / 'valid')) as f:
            file_list_val = [data_path / l.rstrip() for l in f if l.rstrip()]
        cached_files_val = [cache / (f.name.rsplit('.')[0] + '.npy') for f in file_list_val] if cache else None
        valid_data = SVDDataset(file_list_val, cached_files_val, lab_file, data_config, None)
    else:
        valid_data = None

    # do training
    run_training(train_data, valid_data, cfg, spec_len, modelfile, options.save_errors)


if __name__ == '__main__':
    main()
