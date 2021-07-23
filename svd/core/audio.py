import torch
import librosa
import warnings
import torchaudio
import numpy as np

from pathlib import Path
from scipy.ndimage import affine_transform
from torchaudio.transforms import Spectrogram
from svd.utils.labels import create_aligned_targets
from torch.utils.data import IterableDataset, Dataset
torchaudio.set_audio_backend('sox_io')


def load_spec(file_idx, cached_files, files, sample_rate, spec_trf):
    """ Loads cached spectrogram, or else raw audio and transforms it to spectrogram. """
    # loads (cached) files
    if cached_files and cached_files[file_idx].exists():
        return torch.tensor(np.load(str(cached_files[file_idx])))
    else:
        if not files[file_idx].exists() and files[file_idx].with_suffix('.wav').exists():
            fl = str(files[file_idx].with_suffix('.wav'))
        else:
            fl = str(files[file_idx])
        # file is not yet cached
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data, sr = librosa.load(str(fl), sr=sample_rate)
            data = torch.tensor(data)
            spec = spec_trf.forward(data).transpose(-2, -1)
        if cached_files:
            np.save(cached_files[file_idx], spec.numpy())
        return spec


class IterSVDDataset(IterableDataset):
    """ Iterable Dataset for Singing-voice-detector data (useful for training). """
    def __init__(self, files, cached_files, label_path, config):
        super().__init__()
        self.files = files
        self.cached_files = cached_files
        self.label_path = Path(label_path)

        if self.cached_files and len(self.files) != len(self.cached_files):
            raise ValueError('Amount of cached files and regular files is not equal!')

        # prepare spec-transforms
        self.fps = config['fps']
        self.sample_rate = config['sample_rate']
        hop_size = self.sample_rate // self.fps
        self.spec_trf = Spectrogram(n_fft=config['frame_len'], hop_length=hop_size, power=1.)

        # prepare for random excerpts
        self.bins = config['bins']
        self.frames = config['frames']
        self.batch_size = config['batch_size']
        self.indices, self.labels = self.prep_indices_and_labels()

        # prepare random stretch / shift
        self.augment = config['augment']
        self.max_stretch = config['max_stretch']
        self.max_shift = config['max_shift']
        self.keep_frames = config['keep_frames']
        self.keep_bins = config['keep_bins']
        self.order = config['order']

        # prepare random filtering
        self.max_freq = config['max_freq']
        self.max_db = config['max_db']
        self.min_std = 5
        self.max_std = 7

    def prep_indices_and_labels(self):
        # prepare for iterations, collect array of all possible (spect_idx, frame_idx) combinations
        indices = []
        labels = []
        bin_shape = load_spec(0, self.cached_files, self.files, self.sample_rate, self.spec_trf).shape[1]
        if not self.bins or bin_shape < self.bins:
            self.bins = bin_shape
        for spec_idx in range(len(self.files)):
            # get spec
            spec = load_spec(spec_idx, self.cached_files, self.files, self.sample_rate, self.spec_trf)
            indices.append(np.vstack((np.ones(len(spec) - self.frames + 1, dtype=np.int) * spec_idx,
                                      np.arange(len(spec) - self.frames + 1, dtype=np.int))).T)
            # get label
            label_file = self.label_path / (self.files[spec_idx].name.rsplit('.', 1)[0] + '.lab')
            with open(label_file) as f:
                segments = [l.rstrip().split() for l in f if l.rstrip()]
            segments = [(float(start), float(end), label == 'sing') for start, end, label in segments]
            timestamps = torch.arange(len(spec)) / float(self.fps)
            labels.append(create_aligned_targets(segments, timestamps, torch.bool))
        indices = np.vstack(indices)
        return indices, labels

    def apply_random_stretch_shift(self, specs):
        """
        Taken from: https://github.com/f0k/ismir2015/blob/phd_extra/experiments/augment.py;
        Apply random time stretching of up to +/- `max_stretch`, random pitch
        shifting of up to +/- `max_shift`, keeping the central `keep_frames` frames
        and the first `keep_bins` bins.
        """
        outputs = np.empty((len(specs), self.keep_frames, self.keep_bins), dtype=specs.numpy().dtype)
        randoms = (np.random.rand(len(specs), 2) - .5) * 2
        for spec, output, random in zip(specs, outputs, randoms):
            stretch = 1 + random[0] * self.max_stretch
            shift = 1 + random[1] * self.max_shift
            offset = (.5 * (len(spec) - self.keep_frames / stretch), 0)
            # We can do shifting/stretching and cropping in a single affine
            # transform (including setting the upper bands to zero if we shift
            # down the signal so far that we exceed the input's nyquist rate)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                affine_transform(spec.numpy(), (1 / stretch, 1 / shift),
                                 output_shape=(self.keep_frames, self.keep_bins), output=output, offset=offset,
                                 mode='constant', cval=0, order=self.order, prefilter=True)
        # clip possible negative values introduced by the interpolation
        np.maximum(outputs, 0, outputs)
        return torch.tensor(outputs)

    def apply_random_filters(self, specs):
        """
        Taken from: https://github.com/f0k/ismir2015/blob/phd_extra/experiments/augment.py;
        Applies random filter responses to logarithmic-magnitude mel spectrograms.
        The filter response is a Gaussian of a standard deviation between `min_std`
        and `max_std` semitones, a mean between 150 Hz and `max_freq`, and a
        strength between -/+ `max_db` dezibel. Assumes the spectrograms cover up to
        `max_freq` Hz.
        """
        batch_size, length, bins = specs.shape
        # sample means and std deviations on logarithmic pitch scale
        min_pitch = 12 * np.log2(150)
        max_pitch = 12 * np.log2(self.batch_size)
        mean = min_pitch + (np.random.rand(batch_size) * (max_pitch - min_pitch))
        std = self.min_std + np.random.randn(batch_size) * (self.max_std - self.min_std)
        # convert means and std deviations to linear frequency scale
        std = 2 ** ((mean + std) / 12) - 2 ** (mean / 12)
        mean = 2 ** (mean / 12)
        # convert means and std deviations to bins
        mean = mean * bins / self.max_freq
        std = std * bins / self.max_freq
        # sample strengths uniformly in dB
        strength = self.max_db * 2 * (np.random.rand(batch_size) - .5)
        # create Gaussians
        filt = (strength[:, np.newaxis] *
                np.exp(np.square((np.arange(bins) - mean[:, np.newaxis]) / std[:, np.newaxis]) * -.5))
        # transform from dB to factors
        filt = 10 ** (filt / 20.)
        # apply (multiply, broadcasting over the second axis)
        filt = np.asarray(filt, dtype=specs.numpy().dtype)
        return torch.tensor(specs.numpy() * filt[:, np.newaxis, :])

    def __iter__(self):
        indices = self.indices
        while True:
            np.random.shuffle(indices)      # shuffle indices
            for batch_idx in range(0, len(indices), self.batch_size):
                cur_indices = indices[batch_idx:batch_idx + self.batch_size, :]
                batch_specs = torch.stack([load_spec(spec_idx, self.cached_files, self.files, self.sample_rate,
                                                     self.spec_trf)[frame_idx:frame_idx + self.frames, :]
                                           for spec_idx, frame_idx in cur_indices])
                batch_labels = torch.stack([self.labels[spec_idx][frame_idx + self.frames // 2]
                                            for spec_idx, frame_idx in cur_indices])
                if self.augment:
                    batch_specs = self.apply_random_stretch_shift(batch_specs)
                    batch_specs = self.apply_random_filters(batch_specs)
                # insert channel dimension, return batch of specs and their labels
                yield batch_specs.unsqueeze(dim=1), batch_labels.unsqueeze(-1)


class SVDDataset(Dataset):
    """ Dataset for Singing-voice-detector iterating over entire spectrograms (useful for testing/validation). """
    def __init__(self, files, cached_files, label_path, config, pitch_shift=None):
        super().__init__()
        self.files = files
        self.cached_files = cached_files
        self.label_path = Path(label_path)
        self.pitch_shift = pitch_shift
        self.mel_max = config['mel_max'] if self.pitch_shift else None

        if self.cached_files and len(self.files) != len(self.cached_files):
            raise ValueError('Amount of cached files and regular files is not equal!')

        # prepare spec-transforms
        self.fps = config['fps']
        self.sample_rate = config['sample_rate']
        hop_size = self.sample_rate // self.fps
        self.spec_trf = Spectrogram(n_fft=config['frame_len'], hop_length=hop_size, power=1.)

    def __len__(self):
        return len(self.files)

    def shift(self, spec):
        """ Performs pitch-shift. """
        spline_order = 2
        spec = spec.numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            affine_transform(spec, (1, 1 / (1 + self.pitch_shift / 100.)),
                             output_shape=(len(spec), self.mel_max), order=spline_order)
        return torch.tensor(spec)

    def __getitem__(self, item):
        """ Returns single (original-length) spectrogram and according labels. """
        # read (cached) spectrogram
        spec = load_spec(item, self.cached_files, self.files, self.sample_rate, self.spec_trf)
        spec = self.shift(spec) if self.pitch_shift else spec
        # get label
        label_file = self.label_path / (self.files[item].name.rsplit('.', 1)[0] + '.lab')
        with open(label_file) as f:
            segments = [l.rstrip().split() for l in f if l.rstrip()]
        segments = [(float(start), float(end), label == 'sing') for start, end, label in segments]
        timestamps = torch.arange(len(spec)) / float(self.fps)
        labels = create_aligned_targets(segments, timestamps, torch.bool)
        # insert channel dimension, return spectrogram and labels
        return spec.unsqueeze(0), labels.unsqueeze(-1)
