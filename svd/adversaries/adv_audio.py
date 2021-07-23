import torch
import librosa
import torchaudio

from pathlib import Path
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram
torchaudio.set_audio_backend('sox_io')


def get_feature(x, config):
    """ Given raw audio and configuration dictionary, computes spectrogram. """
    # prepare spectrogram transform
    fps = config['fps']
    sample_rate = config['sample_rate']
    hop_size = sample_rate // fps
    spec_trf = Spectrogram(n_fft=config['frame_len'], hop_length=hop_size, power=1.).to(x.device)
    # compute spectrogram
    x = spec_trf.forward(x).transpose(-2, -1).unsqueeze(0)
    padding = torch.zeros((config['blocklen'] // 2, x.shape[-1]), dtype=torch.float)[None, None, ...].to(x.device)
    # prepare for network
    x = torch.cat((padding, x, padding), dim=-2)
    return x


class RawSVDDataset(Dataset):
    """ Dataset for Singing-voice-detector, returning raw audio files for adversarial attack on waveforms. """
    def __init__(self, files, label_path, sample_rate):
        super().__init__()
        self.files = files
        self.label_path = Path(label_path)
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.files)

    def load_audio(self, file_idx):
        """ Loads raw audio. """
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data, sr = librosa.load(str(self.files[file_idx]), sr=self.sample_rate)
        data = torch.tensor(data)
        return data

    def __getitem__(self, item):
        """ Returns loaded audio an appropriate labels. """
        # load audio
        data = self.load_audio(item)
        # get label
        label_file = self.label_path / (self.files[item].name.rsplit('.', 1)[0] + '.lab')
        with open(label_file) as f:
            segments = [l.rstrip().split() for l in f if l.rstrip()]
        segments = [(float(start), float(end), label == 'sing') for start, end, label in segments]

        return data, segments
