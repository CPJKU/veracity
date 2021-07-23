import torch
import torchaudio

from torch import nn


def cw_loss(logits, target, x, delta, alpha):
    """ C&W loss, adapted from https://github.com/carlini/audio_adversarial_examples/blob/master/attack.py. """
    loss_fun = nn.BCEWithLogitsLoss()
    loss = torch.sum(delta ** 2)
    return loss + alpha * loss_fun(logits, target)


def multi_scale_cw_loss(logits, target, x, delta, alpha):
    """ C&W loss adaptation with using multi-scale loss. """
    # see https://github.com/magenta/ddsp/blob/b7488c5916c8c18a3a262c1e3660b37f38560120/ddsp/losses.py#L31
    loss_fun = nn.BCEWithLogitsLoss()

    fft_sizes = [2048, 1024, 512, 256, 128, 64]
    spectral_loss = 0
    eps = 1e-7
    multi_scale_alpha = 1.

    for n_fft in fft_sizes:
        spec_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=int(0.25 * n_fft),
                                                           window_fn=torch.hann_window, power=1).to(x.device)
        # compute magnitude specs
        spec_1 = torch.abs(spec_transform.forward(x.view(1, -1)))
        spec_2 = torch.abs(spec_transform.forward((x + delta).view(1, -1)))

        log_diff = torch.log(spec_1 + eps) - torch.log(spec_2 + eps)
        diff = torch.mean(torch.abs(spec_1 - spec_2)) + multi_scale_alpha * torch.mean(torch.abs(log_diff))
        spectral_loss += diff

    return spectral_loss + alpha * loss_fun(logits, target)
