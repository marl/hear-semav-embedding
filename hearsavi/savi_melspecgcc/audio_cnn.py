#!/usr/bin/env python3
import torch
from math import factorial
from torchaudio.functional import melscale_fbanks
from hearsavi.savi.audio_cnn import AudioCNN as BaseAudioCNN


def nCr(n, r):
    assert r <= n, "r cannot be greater than n"
    return factorial(n) // (factorial(r) * factorial(n - r))


class AudioCNN(BaseAudioCNN):
    r"""A Simple 3-Conv CNN followed by a fully connected layer

    Takes in observations of audio-sensor and produces an embedding of the spectrograms.

    Args:
        output_size: The size of the embedding vector
    """
    sample_rate = 16000
    num_channels = 2 
    embedding_size = 128
    n_mels = 64
    downsample = None
    hop_length = 320
    win_length = 640
    n_fft = 1024
    include_gcc_phat = True
    scene_embedding_size = embedding_size
    timestamp_embedding_size = embedding_size


    def __init__(self):
        super().__init__(
            cnn_dims=(
                self.n_mels,
                (self.sample_rate // self.hop_length) + 1,
            ),
            # 2 audio channels + nCr(2, 2) GCC channels
            num_input_channels=(self.num_channels + nCr(self.num_channels, 2))
        )
        self.register_buffer('window', torch.hann_window(self.win_length))
        self.register_buffer('mel_scale',
            melscale_fbanks(
                n_freqs=(self.n_fft // 2) + 1,
                f_min=0,
                f_max=self.sample_rate / 2, # nyquist
                n_mels=self.n_mels,
                sample_rate=self.sample_rate,
                mel_scale="htk",
                norm=None,
            )
        )