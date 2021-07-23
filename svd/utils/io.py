import torch
import pickle
import numpy as np

from scipy.io.wavfile import write


def save_config(save_path, config):
    """ Saves configuration dictionary to given save path. """
    config_str = [str(k) + ' = ' + str(config[k]) for k in config.keys()]
    with open(save_path, 'w') as f:
        f.write('\n'.join(config_str))


def save_model(save_path, net):
    """ Saves a (trained) network with pytorch. """
    torch.save(net.state_dict(), save_path)


def load_model(load_path, net, device):
    """ Loads a stored network with pytorch. """
    checkpoint = torch.load(load_path, map_location=device)
    net.load_state_dict(checkpoint)
    net.to(device)
    net.eval()
    return net


def save_errors(save_path, errors):
    """ Saves list of errors at given file-path. """
    np.savez(save_path, np.array(errors))


def save_adversary(perturbation, file_path):
    """ Saves adversarial data. """
    write(str(file_path.with_suffix('.wav')), 22050, perturbation.view(-1, 1).numpy())


def pickle_dump(x, path):
    """ Dumps given data to file path. """
    pickle.dump(x, open(path, "wb"))


def pickle_load(path):
    """ Loads pickled data from given file path. """
    return pickle.load(open(path, "rb"))
