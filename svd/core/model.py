import torch
import torchaudio
import numpy as np
import torch.nn as nn
import torch.nn.functional as fu
import torchaudio.functional as func
torchaudio.set_audio_backend('sox_io')


def create_mel_filterbank(sample_rate, frame_len, num_bands, min_freq, max_freq):
    """
    Taken from https://github.com/f0k/ismir2015/blob/master/experiments/audio.py:
    Creates a mel filterbank of `num_bands` triangular filters, with the first
    filter starting at `min_freq` and the last one stopping at `max_freq`.
    Returns the filterbank as a matrix suitable for a dot product against
    magnitude spectra created from samples at a sample rate of `sample_rate`
    with a window length of `frame_len` samples.
    """
    # prepare output matrix
    input_bins = (frame_len // 2) + 1
    filterbank = np.zeros((input_bins, num_bands))

    # mel-spaced peak frequencies
    min_mel = 1127 * np.log1p(min_freq / 700.0)
    max_mel = 1127 * np.log1p(max_freq / 700.0)
    spacing = (max_mel - min_mel) / (num_bands + 1)
    peaks_mel = min_mel + np.arange(num_bands + 2) * spacing
    peaks_hz = 700 * (np.exp(peaks_mel / 1127) - 1)
    fft_freqs = np.linspace(0, sample_rate / 2., input_bins)
    peaks_bin = np.searchsorted(fft_freqs, peaks_hz)

    # fill output matrix with triangular filters
    for b, filt in enumerate(filterbank.T):
        # The triangle starts at the previous filter's peak (peaks_freq[b]),
        # has its maximum at peaks_freq[b+1] and ends at peaks_freq[b+2].
        left_hz, top_hz, right_hz = peaks_hz[b:b+3]  # b, b+1, b+2
        left_bin, top_bin, right_bin = peaks_bin[b:b+3]
        # Create triangular filter compatible to yaafe
        filt[left_bin:top_bin] = ((fft_freqs[left_bin:top_bin] - left_hz) /
                                  (top_bin - left_bin))
        filt[top_bin:right_bin] = ((right_hz - fft_freqs[top_bin:right_bin]) /
                                   (right_bin - top_bin))
        filt[left_bin:right_bin] /= filt[left_bin:right_bin].sum()

    return torch.tensor(filterbank, dtype=torch.float)


class ZeroMeanConvolution(nn.Conv2d):
    """ Zero-Mean Convolutional Layer. """
    def __init__(self, a, b, **kwargs):
        super(ZeroMeanConvolution, self).__init__(a, b, **kwargs)

    def forward(self, x):
        weight = self.weight - self.weight.mean(axis=(2, 3), keepdim=True)
        return fu.conv2d(x, weight)


class SingingVoiceDetector(nn.Module):
    """ Singing-Voice-Detector Pytorch implementation. """

    def __init__(self, config):
        super(SingingVoiceDetector, self).__init__()
        self.config = config
        self.pre_proc = self.get_pre_processing_layer()
        self.mag_trans = self.get_mag_transform_layer()
        self.freq_norm = nn.BatchNorm1d(self.config['n_mels'], eps=1e-4, affine=False)

        if self.config['arch'] != 'ismir2015' and self.config['arch'] != 'ismir2016':
            raise ValueError('Invalid architecture defined (choose ismir2015 or ismir2016)!')

        # prepare convolutional layers
        first_layer = self.config.get('arch.firstconv_zeromean', 'std')
        if first_layer == '0mean':
            self.conv1 = ZeroMeanConvolution(1, 64, kernel_size=3, stride=1, padding=0)
        elif first_layer == 'std':
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0)
        else:
            raise ValueError('Invalid first_layer {}. Use 0mean or std!'.format(first_layer))

        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=0, dilation=(3, 1))
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0, dilation=(3, 1))
        self.conv_16 = nn.Conv2d(64, 128, kernel_size=(3, 21 - 3), stride=1, padding=0, dilation=(3, 1))

        # prepare batch-norm layers
        bn = self.config['arch.batch_norm']
        self.bn1 = nn.BatchNorm2d(64, eps=1e-4) if bn else lambda x: x
        self.bn2 = nn.BatchNorm2d(32, eps=1e-4) if bn else lambda x: x
        self.bn3 = nn.BatchNorm2d(128, eps=1e-4) if bn else lambda x: x
        self.bn4 = nn.BatchNorm2d(64, eps=1e-4) if bn else lambda x: x
        self.bn_16 = nn.BatchNorm2d(128, eps=1e-4) if bn else lambda x: x
        self.bn_d1 = nn.BatchNorm2d(256, eps=1e-4) if bn else lambda x: x
        self.bn_d2 = nn.BatchNorm2d(64, eps=1e-4) if bn else lambda x: x

        # prepare dropout
        self.do1 = self.get_dropout_layer()
        self.do2 = self.get_dropout_layer()
        self.do3 = self.get_dropout_layer()
        self.do_16 = self.get_dropout_layer()
        self.do_d1 = nn.Dropout(0.5)
        self.do_d2 = nn.Dropout(0.5)
        self.do_d3 = nn.Dropout(0.5)

        # prepare dense layers
        if self.config['arch'] == 'ismir2016':
            self.fc1 = nn.Conv2d(128, 256, (31, 1), dilation=(3, 1))
        else:
            self.fc1 = nn.Conv2d(64, 256, (11, 7), dilation=(9, 1))
        self.fc2 = nn.Conv2d(256, 64, 1)
        self.fc3 = nn.Conv2d(64, 1, 1)

        # prepare sigmoid
        self.sigmoid = nn.Sigmoid()

        self.init()

    def init(self):
        # initialise weights (orthogonally)
        nn.init.orthogonal_(self.conv1.weight)
        nn.init.orthogonal_(self.conv2.weight)
        nn.init.orthogonal_(self.conv3.weight)
        nn.init.orthogonal_(self.conv4.weight)
        nn.init.orthogonal_(self.conv_16.weight)

        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)

    def get_pre_processing_layer(self):
        """ Prepare pre-processing layer (computing Mel transform) for network. """
        if self.config['filterbank'] == 'mel':
            fb = func.create_fb_matrix(self.config['n_stft'], self.config['f_min'], self.config['f_max'],
                                       self.config['n_mels'], self.config['sample_rate'],
                                       norm='slaney')
            return lambda x: torch.matmul(x.reshape(-1, x.shape[-2], x.shape[-1]),
                                          fb.to(x.device)).reshape(x.shape[:-1]
                                                                   + fb.shape[-1:])[:self.config['spec_len']]
        elif self.config['filterbank'] == 'mel_orig':
            fb = create_mel_filterbank(self.config['sample_rate'], self.config['frame_len'], self.config['n_mels'],
                                       self.config['f_min'], self.config['f_max'])
            return lambda x: torch.matmul(x.reshape(-1, x.shape[-2], x.shape[-1]),
                                          fb.to(x.device)).reshape(x.shape[:-1]
                                                                   + fb.shape[-1:])[:self.config['spec_len']]
        elif self.config['filterbank'] != 'none':
            raise ValueError('Invalid filterbank! Choose one of [mel, mel_orig, none]')
        else:
            return lambda x: x

    def get_mag_transform_layer(self):
        """ Prepare layer for magnitude transformation. """
        if self.config['magscale'] == 'log':
            mag_trans = lambda x: torch.log(torch.max(torch.tensor(1e-7).to(x.device), x))
        elif self.config['magscale'] == 'log1p':
            mag_trans = lambda x: torch.log1p(x)
        elif self.config['magscale'] != 'none':
            raise ValueError('Invalid magnitude scaling defined!')
        else:
            mag_trans = lambda x: x

        return mag_trans

    def get_dropout_layer(self):
        if self.config['arch.convdrop'] == 'independent':
            return nn.Dropout(0.1)
        elif self.config['arch.convdrop'] != 'none':
            raise ValueError('Invalid arch.convdrop defined!')
        else:
            return lambda x: x

    def forward(self, x, preprocess=True):
        """ Performs forward step. """
        if preprocess:
            x = self.pre_proc(x)
            x = self.mag_trans(x)

        # reshape x to normalise per frequency band (with batch norm)
        x = self.freq_norm(x.transpose(1, -1).squeeze(dim=-1)).unsqueeze(-1).transpose(1, -1)

        x = self.conv1(x)
        x = self.do1(fu.leaky_relu(self.bn1(x), negative_slope=0.01, inplace=False))
        x = fu.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01, inplace=False)
        x = self.do2(fu.max_pool2d(x, kernel_size=3, stride=(1, 3)))

        x = self.do3(fu.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01, inplace=False))
        x = fu.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01, inplace=False)

        if self.config['arch'] == 'ismir2015':
            x = fu.max_pool2d(x, kernel_size=3, stride=(1, 3), dilation=(3, 1))
        elif self.config['arch'] == 'ismir2016':
            x = self.do_16(x)
            x = fu.leaky_relu(self.bn_16(self.conv_16(x)), negative_slope=0.01, inplace=False)
            x = fu.max_pool2d(x, kernel_size=(1, 4), stride=(1, 4), dilation=(3, 1))

        # x = x.reshape(x.shape[0], -1)
        x = fu.leaky_relu(self.bn_d1(self.fc1(self.do_d1(x))), negative_slope=0.01, inplace=False)
        x = fu.leaky_relu(self.bn_d2(self.fc2(self.do_d2(x))), negative_slope=0.01, inplace=False)
        x = self.fc3(self.do_d3(x))

        return x.view(-1, 1)

    @torch.no_grad()
    def probabilities_from_preprocessed_spec(self, x):
        """ Returns logits after sigmoid without preprocessing. """
        logits = self.forward(x, preprocess=False)
        return self.sigmoid(logits).view(-1)

    @torch.no_grad()
    def probabilities(self, x):
        """ Returns logits after sigmoid. """
        logits = self.forward(x)
        return self.sigmoid(logits).view(-1)

    @torch.no_grad()
    def probabilities_from_logits(self, logits):
        """ Applies sigmoid to given logits. """
        return self.sigmoid(logits).view(-1)
