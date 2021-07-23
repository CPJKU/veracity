import torch

from pathlib import Path
from torch.utils.data import DataLoader

from svd.utils.io import load_model
from svd.core.audio import SVDDataset
from svd.train import prep_net_config
from svd.utils.progress import progress
from svd.core.model import SingingVoiceDetector
from svd.adversaries.raw_attack import opts_parser, prep_logger, prep_perturbation, read_configs, \
    update, get_prediction, check_convergence


def get_data_config(config):
    """ Prepares configuration dictionary for validation dataset. """
    data_config = {}
    bin_nyquist = config['frame_len'] // 2 + 1
    bin_mel_max = bin_nyquist * 2 * config['mel_max'] // config['sample_rate']
    data_config.update({'fps': config['fps'],
                        'sample_rate': config['sample_rate'],
                        'frame_len': config['frame_len'],
                        'mel_max': config['mel_max'],
                        'mel_min': config['mel_min'],
                        'n_mels': config['mel_bands'],
                        'blocklen': config['blocklen'],
                        'spec_len': bin_mel_max})
    return data_config


def save_spec_adv(adv_spec, save_path):
    """ Saves adversarial spectrogram with numpy. """
    import numpy as np
    np.save(Path(save_path).with_suffix('.npy'), adv_spec.cpu().numpy())


def write_log_file(logger, cur_file, orig_preds, mod_preds, cors, gts, ad_snr):
    """ Handles log-file writing for each frame within an audio file. """
    for frame, (orig, mod, c, g) in enumerate(zip(orig_preds, mod_preds, cors, gts)):
        # write to log file
        logger.append([cur_file, frame, g, orig, mod, ad_snr, c])


def get_adv_perturbation(x, label, net, adv_config, threshold):
    """ Computes adversarial perturbation for a specific example. """
    # prepare perturbation and optimisation
    device = x.device
    delta, optimiser, loss_fun = prep_perturbation(x.shape, device, adv_config['lr'], adv_config['attack'])
    cur_snr, mod_pred, conv, cor = 0., None, False, []

    # get original prediction
    orig_pred = get_prediction(x, net, 'high', None, threshold)
    print('Original correct predictions: {}\n'.format((orig_pred == label.squeeze()).sum()))
    target = (~orig_pred).to(device)

    for i in range(1, adv_config['max_iterations'] + 1):
        # perform update step
        update(x + delta, x, delta, target, net, optimiser, loss_fun, adv_config)
        # check new prediction / convergence
        mod_pred, conv, cor = check_convergence(x + delta, net, target, 'high', None,
                                                threshold, adv_config['req_change_factor'])
        # instead of snr, use norm of delta for now
        cur_snr = torch.sum(delta ** 2).item()
        print('\rep {}/{}; current cor: {}/{}; current norm: {}'.format(i, adv_config['max_iterations'],
                                                                        cor.sum().item(), len(cor), cur_snr),
              flush=True, end='')
        if conv:
            return orig_pred, mod_pred, label, cur_snr, cor, delta

    return orig_pred, mod_pred, label, cur_snr, cor, delta


def run_attack(config, adv_config, data, model_path, data_config, adversarial_path, threshold):
    """ Main method for adversarial attack. """
    # prepare network
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net_config = prep_net_config(config, data_config['spec_len'])
    net = load_model(model_path, SingingVoiceDetector(net_config), device)

    # prepare logger
    logger = prep_logger(adversarial_path / 'logger.csv')

    # prepare padding
    padding = torch.zeros((config['blocklen'] // 2, config['frame_len'] // 2 + 1), dtype=torch.float)[None, None, ...]
    padding = padding.to(device)

    for i, (x, y) in enumerate(progress(data, desc='Corrupting file ')):
        # compute adversarial perturbation, log progress
        x, y = x.to(device), y.to(device)
        x = torch.cat((padding, x, padding), dim=-2)
        orig_pred, mod_pred, gt, ad_snr, cor, delta = get_adv_perturbation(x, y, net, adv_config, threshold)
        cur_file = data.dataset.files[i].name

        # save to log file
        write_log_file(logger, cur_file, orig_pred.cpu().numpy(), mod_pred.cpu().numpy(),
                       cor.numpy(), gt.squeeze().cpu().numpy(), ad_snr)

        # if successful, save adversary
        if cor.sum() > 0:
            save_spec_adv((x + delta.detach()).cpu(), adversarial_path / cur_file)


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    modelfile = options.modelfile

    # get configurations
    adv_cfg, adversarial_path, cfg = read_configs(modelfile, options)
    adv_cfg['attack'] = 'cw'

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
    data_config = get_data_config(cfg)
    data = SVDDataset(file_list, None, label_path, data_config, None)
    data = DataLoader(data, batch_size=1, shuffle=False)

    # run attack
    run_attack(cfg, adv_cfg, data, modelfile, data_config, adversarial_path, options.threshold)


if __name__ == '__main__':
    main()
