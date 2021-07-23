import torch


def create_aligned_targets(segments, timestamps, dtype=torch.float32):
    """ Align labels to spectrogram, taken from https://github.com/f0k/ismir2015/blob/master/experiments/labels.py. """
    targets = torch.zeros(len(timestamps), dtype=dtype)
    if not segments:
        return targets
    starts, ends, labels = zip(*segments)
    starts = torch.searchsorted(timestamps.squeeze(), torch.tensor(starts))
    ends = torch.searchsorted(timestamps.squeeze(), torch.tensor(ends))
    for a, b, l in zip(starts, ends, labels):
        targets[a:b] = l
    return targets
