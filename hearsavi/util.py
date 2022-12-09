"""
Utility functions for hear-kit
"""

from types import ModuleType
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch import Tensor


def compute_spectrogram(
    audio_data,
    win_length: int,
    hop_length: int,
    n_fft: int,
    n_mels: int,
    window: Optional[torch.Tensor],
    mel_scale: Optional[torch.Tensor],
    downsample: Optional[int],
    include_gcc_phat: bool,
    backend: str = "torch",
):
    assert backend in ("torch", "numpy")
    # stft.shape = (C=2, T, F)
    stft = torch.stack(
        [
            torch.stft(
                input=torch.tensor(X_ch, device='cpu', dtype=torch.float32),
                win_length=win_length,
                hop_length=hop_length,
                n_fft=n_fft,
                center=True,
                window=(
                    window if window is not None
                    else torch.hann_window(win_length, device="cpu")
                ),
                pad_mode="constant", # constant for zero padding
                return_complex=True,
            ).T
            for X_ch in audio_data
        ],
        dim=0
    )
    # Compute power spectrogram
    spectrogram = (torch.abs(stft) ** 2.0).to(dtype=torch.float32)
    # Apply the mel-scale filter to the power spectrogram
    if mel_scale is not None:
        spectrogram = torch.matmul(spectrogram, mel_scale)
    # Optionally downsample
    if downsample:
        spectrogram = torch.nn.functional.avg_pool2d(
            spectrogram.unsqueeze(dim=0),
            kernel_size=(downsample, downsample),
        ).squeeze(dim=0)
    # Convert to decibels
    spectrogram = F.amplitude_to_DB(
        spectrogram,
        multiplier=20.0,
        amin=1e-10,
        db_multiplier=0.0,
        top_db=80,
    )

    if include_gcc_phat:
        num_channels = stft.shape[0]
        n_freqs = n_mels if (mel_scale is not None) else ((n_fft // 2) + 1)
        # compute gcc_phat : (comb, T, F)
        out_list = []
        for ch1 in range(num_channels - 1):
            for ch2 in range(ch1 + 1, num_channels):
                x1 = stft[ch1]
                x2 = stft[ch2]
                xcc = torch.angle(x1 * torch.conj(x2))
                xcc = torch.exp(1j * xcc.type(torch.complex64))
                gcc_phat = torch.fft.irfft(xcc)
                # Just get a subset of GCC values to match dimensionality
                gcc_phat = torch.cat(
                    [
                        gcc_phat[..., -n_freqs // 2:],
                        gcc_phat[..., :n_freqs // 2],
                    ],
                    dim=-1,
                )
                out_list.append(gcc_phat)
        gcc_phat = torch.stack(out_list, dim=0)

        # Downsample
        if downsample:
            gcc_phat = torch.nn.functional.avg_pool2d(
                gcc_phat,
                kernel_size=(downsample, downsample),
            )

        # spectrogram.shape = (C=3, T, F)
        spectrogram = torch.cat([spectrogram, gcc_phat], dim=0)

    # Reshape to how SoundSpaces expects
    # spectrogram.shape = (F, T, C)
    spectrogram = spectrogram.permute(2, 1, 0)
    if backend == "torch":
        return spectrogram
    elif backend == "numpy":
        return spectrogram.numpy().astype(np.float32)
    

def frame_audio(
    audio: Tensor, frame_size: int, hop_size: float, sample_rate: int
) -> Tuple[Tensor, Tensor]:
    """
    Slices input audio into frames that are centered and occur every
    sample_rate * hop_size samples. We round to the nearest sample.

    Args:
        audio: input audio, expects a 3d Tensor of shape:
            (n_sounds, num_channels, num_samples)
        frame_size: the number of samples each resulting frame should be
        hop_size: hop size between frames, in milliseconds
        sample_rate: sampling rate of the input audio

    Returns:
        - A Tensor of shape (n_sounds, num_channels, num_frames, frame_size)
        - A Tensor of timestamps corresponding to the frame centers with shape:
            (n_sounds, num_frames).
    """

    # Zero pad the beginning and the end of the incoming audio with half a frame number
    # of samples. This centers the audio in the middle of each frame with respect to
    # the timestamps.
    audio = F.pad(audio, (frame_size // 2, frame_size - frame_size // 2))
    num_padded_samples = audio.shape[-1]

    frame_step = hop_size / 1000.0 * sample_rate
    frame_number = 0
    frames = []
    timestamps = []
    frame_start = 0
    frame_end = frame_size
    while True:
        frames.append(audio[:, :, frame_start:frame_end])
        timestamps.append(frame_number * frame_step / sample_rate * 1000.0)

        # Increment the frame_number and break the loop if the next frame end
        # will extend past the end of the padded audio samples
        frame_number += 1
        frame_start = int(round(frame_number * frame_step))
        frame_end = frame_start + frame_size

        if not frame_end <= num_padded_samples:
            break

    # Expand out the timestamps to have shape (n_sounds, num_frames)
    timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)
    timestamps_tensor = timestamps_tensor.expand(audio.shape[0], -1)

    return torch.stack(frames, dim=2), timestamps_tensor