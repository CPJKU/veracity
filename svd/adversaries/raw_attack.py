import torch

from torch import optim
from pathlib import Path
from shutil import copyfile
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from svd.train import prep_net_config
from svd.utils.progress import progress
from svd.adversaries.utils import Logger, snr
from svd.core.model import SingingVoiceDetector
from svd.utils.labels import create_aligned_targets
from svd.utils.io import load_model, save_adversary
from svd.adversaries.adv_audio import RawSVDDataset, get_feature
from svd.adversaries.adv_losses import cw_loss, multi_scale_cw_loss
from svd.utils.config import parse_config_file, parse_variable_assignments


def opts_parser():
    descr = 'Computes adversarial examples for singing voice detector.'
    parser = ArgumentParser(description=descr)
    parser.add_argument('modelfile', metavar='MODELFILE', type=str,
                        help='File to load the learned weights from (.pt format)')
    parser.add_argument('--adv-dir', metavar='DIR', type=str, required=True,
                        help='Directory to save adversaries to.')
    parser.add_argument('--dataset', type=str, default='jamendo',
                        help='Name of the dataset to use (default: %(default)s)')
    parser.add_argument('--dataset-path', metavar='DIR', type=str, default=None, required=True,
                        help='Path to data of dataset which will be used')
    parser.add_argument('--test', action='store_true', default=False,
                        help='If True, uses test-data instead of validation data for computation.')
    parser.add_argument('--mem-use', type=str, choices=('high', 'low'), default='high',
                        help='How much memory to use for the prediction. (default: %(default)s)')
    parser.add_argument('--threshold', type=float, default=None, required=True,
                        help='Threshold for defining what is classified as singing voice.')
    parser.add_argument('--vars', metavar='FILE', action='append', type=str,
                        default=[Path(__file__).parent.parent / 'default.vars'],
                        help='Reads configuration variables from a FILE of KEY=VALUE '
                             'lines. Can be given multiple times, settings from later '
                             'files overriding earlier ones. Will read defaults.vars, then files given here.')
    parser.add_argument('--adv-vars', metavar='FILE', action='append', type=str,
                        default=[Path(__file__).parent / 'adv_default.vars'],
                        help='Reads configuration variables from a FILE of KEY=VALUE '
                             'lines for adversarial attack.')
    parser.add_argument('--var', metavar='KEY=VALUE', action='append', type=str,
                        help='Set the configuration variable KEY to VALUE. Overrides ' 
                             'settings from --vars options. Can be given multiple times.')
    return parser


def read_configs(model_file, options):
    """ Reads (adversarial) configuration files and prepares adversarial save-path. """
    # read configuration of model
    if Path(model_file + '.vars').exists():
        options.vars.insert(1, model_file + '.vars')
    cfg = {}
    for fn in options.vars:
        cfg.update(parse_config_file(fn))
    cfg.update(parse_variable_assignments(options.var))

    # read configuration for attack
    adv_cfg = {}
    adv_cfg.update(parse_config_file(options.adv_vars[0]))
    adversarial_path = Path(options.adv_dir) / adv_cfg['experiment']
    if not adversarial_path.exists():
        adversarial_path.mkdir(parents=True)
    # save current adversarial configuration
    copyfile(options.adv_vars[0], adversarial_path / 'config.vars')
    return adv_cfg, adversarial_path, cfg


def get_data_config(config, spec_len):
    """ Prepares configuration dictionary for validation dataset. """
    data_config = {}
    data_config.update({'fps': config['fps'],
                        'sample_rate': config['sample_rate'],
                        'frame_len': config['frame_len'],
                        'mel_min': config['mel_min'],
                        'mel_max': config['mel_max'],
                        'blocklen': config['blocklen'],
                        'batchsize': config['batchsize'],
                        'n_mels': config['mel_bands'],
                        'spec_len': spec_len})
    return data_config


def prep_perturbation(x_shape, device, lr, attack):
    """ Prepares delta and functions for optimising adversarial perturbation. """
    delta = torch.zeros(x_shape).to(device)
    delta.requires_grad = True
    optimiser = optim.Adam([delta], lr=lr)
    loss_fun = cw_loss if attack.lower() == 'cw' else multi_scale_cw_loss
    return delta, optimiser, loss_fun


def prep_logger(path):
    """ Prepares logging file for adversarial attack. """
    logger = Logger(path, columns=['file', 'frame', 'ground truth', 'orig pred', 'mod pred', 'snr', 'convergences'])
    return logger


def get_label(y, time_len, fps, device):
    """ Aligns labels to spectrogram, prepares target for attack. """
    labels = create_aligned_targets(y, torch.arange(time_len) / float(fps), torch.bool).unsqueeze(-1)
    labels = labels.to(device)
    return labels


def update(in_net, x, delta, target, net, optimiser, loss_fun, config):
    """ Performs single update step that minimises adversarial loss function. """
    # get predictions of net
    logits = net(in_net)
    # compute loss and optimise
    optimiser.zero_grad()
    loss = loss_fun(logits.squeeze(), target.float(), x, delta, config['alpha'])
    loss.backward()
    if config['sign']:
        with torch.no_grad():
            delta.grad /= torch.abs(delta.grad)
            delta.grad[delta.grad != delta.grad] = 0.  # get rid of NANs
    optimiser.step()

    with torch.no_grad():
        if config['clipping']:
            delta.clamp_(min=-config['clip_eps'], max=config['clip_eps'])


def check_convergence(in_net, net, target, mem_use, config, threshold, req_change_factor):
    """ Computes current prediction of potential adversary and checks convergence. """
    # get current prediction
    pred = get_prediction(in_net, net, mem_use, config, threshold).cpu()

    # check which predictions we already changed successfully
    correct = (pred == target.cpu().squeeze())
    conv = correct.sum().item() >= len(correct) * req_change_factor

    return pred, conv, correct


def get_prediction(in_net, net, mem_use, config, threshold):
    """ Gets predictions for given input, network and memory usage. """
    if mem_use == 'high':
        # pass full spectrogram through network
        pred = net(in_net)
    else:
        # pass overlapping snippets (like during training) through network
        block_len = config['blocklen']
        num_excerpts = in_net.shape[-2] - block_len + 1
        excerpts = torch.as_strided(in_net.squeeze(), size=(num_excerpts, block_len, in_net.shape[-1]),
                                    stride=(in_net.shape[-1], in_net.shape[-1], 1)).unsqueeze(1)
        pred = torch.vstack([net(excerpts[pos:pos + config['batchsize'], ...])
                             for pos in range(0, num_excerpts, config['batchsize'])])

    # get probabilities with sigmoid
    pred = net.sigmoid(pred)
    pred = pred.squeeze()
    # apply threshold for predictions
    pred = pred > threshold

    return pred


def get_adv_perturbation(x, y, net, adv_config, mem_use, threshold, config):
    """ Computes adversarial perturbation for a specific example. """
    # prepare perturbation and optimisation
    device = x.device
    delta, optimiser, loss_fun = prep_perturbation(x.shape, device, adv_config['lr'], adv_config['attack'])
    cur_snr, mod_pred, conv, cor = 0., None, False, []

    # get original prediction, align labels and prepare target
    in_net = get_feature(x, config).to(device)
    label = get_label(y, in_net.shape[-2] - config['blocklen'] // 2 * 2, config['fps'], device)

    orig_pred = get_prediction(in_net, net, mem_use, config, threshold)
    print('Original correct predictions: {}\n'.format((orig_pred == label.squeeze()).sum()))
    target = (~orig_pred).to(device)

    for i in range(1, adv_config['max_iterations'] + 1):
        # perform update step
        in_net = get_feature(x + delta, config).to(device)
        update(in_net, x, delta, target, net, optimiser, loss_fun, adv_config)
        # check new prediction / convergence
        new_in_net = get_feature(x + delta, config).to(device)
        mod_pred, conv, cor = check_convergence(new_in_net, net, target, mem_use, config,
                                                threshold, adv_config['req_change_factor'])
        # compute current SNR
        cur_snr = snr(x.cpu().numpy(), (x + delta).cpu().detach().numpy())
        print('\rep {}/{}; current cor: {}/{}; current snr: {}'.format(i, adv_config['max_iterations'],
                                                                       cor.sum().item(), len(cor), cur_snr),
              flush=True, end='')
        if conv:
            return orig_pred, mod_pred, label, cur_snr, cor, delta

    return orig_pred, mod_pred, label, cur_snr, cor, delta


def write_log_file(logger, cur_file, orig_preds, mod_preds, cors, gts, ad_snr):
    """ Handles log-file writing for each frame within an audio file. """
    for frame, (orig, mod, c, g) in enumerate(zip(orig_preds, mod_preds, cors, gts)):
        # write to log file
        logger.append([cur_file, frame, g, orig, mod, ad_snr, c])


def run_attack(config, adv_config, data, model_path, spec_len, adversarial_path, mem_use, threshold):
    """ Main method for adversarial attack. """
    # prepare network
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net_config = prep_net_config(config, spec_len)
    net = load_model(model_path, SingingVoiceDetector(net_config), device)
    data_config = get_data_config(config, spec_len)

    # prepare logger
    logger = prep_logger(adversarial_path / 'logger.csv')

    for i, (x, y) in enumerate(progress(data, desc='Corrupting file ')):
        # compute adversarial perturbation, log progress
        x = x.to(device)
        orig_pred, mod_pred, gt, ad_snr, cor, delta = get_adv_perturbation(x, y, net, adv_config, mem_use,
                                                                           threshold, data_config)
        cur_file = data.dataset.files[i].name

        # save to log file
        write_log_file(logger, cur_file, orig_pred.cpu().numpy(), mod_pred.cpu().numpy(),
                       cor.numpy(), gt.squeeze().cpu().numpy(), ad_snr)

        # if successful, save adversary
        if cor.sum() > 0:
            save_adversary((x + delta.detach()).cpu(), adversarial_path / cur_file)


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    modelfile = options.modelfile

    # get configurations
    adv_cfg, adversarial_path, cfg = read_configs(modelfile, options)

    # prepare paths and files
    data_path = Path(options.dataset_path)
    info_path = Path(__file__).parent.parent.parent / 'dataset' / options.dataset
    label_path = info_path / 'labels'
    if not options.test:
        with open(str(info_path / 'filelists' / 'valid')) as f:
            file_list = [data_path / l.rstrip() for l in f if l.rstrip()]
    else:
        with open(str(info_path / 'filelists' / 'test')) as f:
            file_list = [data_path / l.rstrip() for l in f if l.rstrip()]

    # prepare data
    data = RawSVDDataset(file_list, label_path, cfg['sample_rate'])
    data = DataLoader(data, batch_size=1, shuffle=False)
    bin_nyquist = cfg['frame_len'] // 2 + 1
    bin_mel_max = bin_nyquist * 2 * cfg['mel_max'] // cfg['sample_rate']

    run_attack(cfg, adv_cfg, data, modelfile, bin_mel_max, adversarial_path, options.mem_use, options.threshold)


if __name__ == '__main__':
    main()
