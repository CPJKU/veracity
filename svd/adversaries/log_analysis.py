import csv
import torch
import librosa
import numpy as np

from pathlib import Path
from svd.adversaries.adv_audio import get_feature


def str_to_bool(str_bool):
    """ Helper function to convert string to boolean. """
    if str_bool == "False":
        return False
    return True


def check_convergence(log_file):
    """ Given a log file, returns how much frames per file converged and he overall convergence of an adversary. """
    # read log file
    columns, log = read_log(log_file)

    # prepare indices
    file_idx, = np.where(columns == 'file')[0]
    conv_idx, = np.where(columns == 'convergences')[0]

    # count convergences for each frame of a file
    files = np.unique([l[file_idx] for l in log])
    conv_dict = {k: [0, 0] for k in files}

    for entry in log:
        if entry[conv_idx] == 'True':
            conv_dict[entry[file_idx]][0] += 1
        conv_dict[entry[file_idx]][1] += 1

    tot_convs = np.sum([conv_dict[k][0] for k in conv_dict.keys()])
    all_frames = np.sum([conv_dict[k][1] for k in conv_dict.keys()])

    return conv_dict, tot_convs / all_frames


def _get_data_config():
    """ Hardcoding of data-configuration for evaluation. """
    data_config = {}
    data_config.update({'fps': 70,
                        'sample_rate': 22050,
                        'frame_len': 1024,
                        'blocklen': 115})
    return data_config


def read_log(log_file_path):
    """ Reads given log file, returns columns of log and entries. """
    with open(log_file_path, 'r') as fp:
        reader = csv.reader(fp, delimiter=',')
        log = np.array([l for l in reader])
    columns = log[0]
    log = log[1:]
    return columns, log


def read_file(file, sample_rate):
    """ Reads audio file with librosa. """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data, sr = librosa.load(str(file), sr=sample_rate)

    return torch.tensor(data)


def get_successful_snippets(file):
    """ Given an adversarial file, returns all snippets (with frame-length of 115) that are adversarial. """
    # prepare paths, configurations
    log_file = Path(file.parent) / 'logger.csv'
    data_config = _get_data_config()

    # read log file
    columns, log = read_log(log_file)

    # prepare indices
    file_idx, = np.where(columns == 'file')[0]
    frame_idx, = np.where(columns == 'frame')[0]
    conv_idx, = np.where(columns == 'convergences')[0]
    true_idx,  = np.where(columns == 'ground truth')[0]
    orig_idx, = np.where(columns == 'orig pred')[0]
    modi_idx, = np.where(columns == 'mod pred')[0]

    print('Current file: {}'.format(file.name))
    if file.suffix == '.npy':
        # spectrogram adversaries
        spec = torch.tensor(np.load(file))
    elif file.suffix == '.wav':
        # waveform adversaries: read file, compute spectrogram
        data = read_file(file, data_config['sample_rate'])
        spec = get_feature(data.unsqueeze(0), data_config)

    # get relevant entries in log file (different from ground truth AND original prediction)
    log[:, file_idx] = np.array([n.rsplit('.')[0] for n in log[:, file_idx]])
    cur_log = log[log[:, file_idx] == file.name.rsplit('.')[0], :]
    frames = cur_log[cur_log[:, conv_idx] == 'True', frame_idx]

    for frame in frames:
        frame = int(frame)
        cur_spec = spec[..., frame:frame + 115, :]

        yield cur_spec, cur_log[frame, true_idx], cur_log[frame, orig_idx], cur_log[frame, modi_idx], frame
