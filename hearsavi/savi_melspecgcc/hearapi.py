"""
SAVI audio subnetwork model for HEAR.

Adapted from
https://github.com/marl/sound-spaces/tree/main/ss_baselines/savi
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.fft
from torch import Tensor
from torchaudio.functional import amplitude_to_DB

from hearsavi.util import frame_audio
from hearsavi.savi_melspecgcc.audio_cnn import AudioCNN

# Default hop_size in milliseconds
TIMESTAMP_HOP_SIZE = 50
SCENE_HOP_SIZE = 250
n_fft = 512

# Number of frames to batch process for timestamp embeddings
BATCH_SIZE = 512


def compute_spectrogram(
    audio_data,
    win_length: int,
    hop_length: int,
    n_fft: int,
    n_mels: int,
    window: Optional[torch.Tensor],
    mel_scale: Optional[torch.Tensor],
    downsample: Optional[int],
    include_gcc_phat: bool
):
    # audio_data.shape = (audio_batches, num_channels, num_frames, frame_size)
    num_sounds, num_channels, num_frames, frame_size = audio_data.shape
    stft = torch.stft(
        input=audio_data.reshape(-1, audio_data.shape[-1]),
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
    )
    stft = stft.reshape(num_sounds, num_channels, num_frames, *stft.shape[-2:])
    # stft.shape = (audio_batches, num_channels, num_frames, num_freq, num_time)
    # Compute power spectrogram
    spectrogram = (torch.abs(stft) ** 2.0).to(dtype=torch.float32)
    # Apply the mel-scale filter to the power spectrogram
    if mel_scale is not None:
        spectrogram = torch.matmul(
            spectrogram.transpose(-1, -2),
            mel_scale,
        ).transpose(-1, -2)
    # Optionally downsample
    if downsample:
        spectrogram = torch.nn.functional.avg_pool2d(
            spectrogram,
            kernel_size=(downsample, downsample),
        )
    # Convert to decibels
    spectrogram = amplitude_to_DB(
        spectrogram,
        multiplier=20.0,
        amin=1e-10,
        db_multiplier=0.0,
        top_db=80,
    )
    # stft.shape = (audio_batches, num_channels, num_frames, num_freq, num_time)

    if include_gcc_phat:
        n_freqs = n_mels if (mel_scale is not None) else ((n_fft // 2) + 1)
        # compute gcc_phat : (comb, T, F)
        # compute gcc_phat : (audio_batches, nCr(num_channels, 2), num_frames, num_freq, num_time)
        out_list = []
        for ch1 in range(num_channels - 1):
            for ch2 in range(ch1 + 1, num_channels):
                x1 = stft[:, ch1, ...]
                x2 = stft[:, ch2, ...]
                xcc = torch.angle(x1 * torch.conj(x2))
                xcc = torch.exp(1j * xcc.type(torch.complex64))
                gcc_phat = torch.fft.irfft(xcc, dim=-2)
                # Just get a subset of GCC values to match dimensionality
                gcc_phat = torch.cat(
                    [
                        gcc_phat[..., -n_freqs // 2:, :],
                        gcc_phat[..., :n_freqs // 2, :],
                    ],
                    dim=-2,
                )
                out_list.append(gcc_phat)
        gcc_phat = torch.stack(out_list, dim=1)

        # Downsample
        if downsample:
            gcc_phat = torch.nn.functional.avg_pool2d(
                gcc_phat,
                kernel_size=(downsample, downsample),
            )

        spectrogram = torch.cat([spectrogram, gcc_phat], dim=1)

    # Reshape to how SoundSpaces expects
    # return shape (audio_batches, num_frames, num_channels, num_freq, num_time)
    return spectrogram.transpose(1, 2)
    

def load_model(model_file_path: str = "") -> torch.nn.Module:
    """
    Returns a torch.nn.Module that produces embeddings for audio.

    Args:
        model_file_path: Ignored.
    Returns:
        Model
    """

    audio_model = AudioCNN()

    if model_file_path != "":
        loaded_model = torch.load(model_file_path, map_location=torch.device('cpu'))
        audio_model.load_state_dict(loaded_model)

    return audio_model


def get_timestamp_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
    hop_size: float = TIMESTAMP_HOP_SIZE,
) -> Tuple[Tensor, Tensor]:
    """
    This function returns embeddings at regular intervals centered at timestamps. Both
    the embeddings and corresponding timestamps (in milliseconds) are returned.

    Args:
        audio: n_sounds x n_channels x n_samples of audio in the range [-1, 1].
        model: Loaded model.
        hop_size: Hop size in milliseconds.
            NOTE: Not required by the HEAR API. We add this optional parameter
            to improve the efficiency of scene embedding.

    Returns:
        - Tensor: embeddings, A float32 Tensor with shape (n_sounds, n_timestamps,
            model.timestamp_embedding_size).
        - Tensor: timestamps, Centered timestamps in milliseconds corresponding
            to each embedding in the output. Shape: (n_sounds, n_timestamps).
    """

    # Assert audio is of correct shape
    if audio.ndim != 3:
        raise ValueError(
            "audio input tensor must be 3D with shape (n_sounds, num_channels, num_samples)"
        )

    if audio.shape[1] != 2:
        raise ValueError(
            "audio input tensor must be binaural"
        )

    # Make sure the correct model type was passed in
    if not isinstance(model, AudioCNN):
        raise ValueError(
            f"Model must be an instance of {AudioCNN.__name__}"
        )

    # Send the model to the same device that the audio tensor is on.
    model = model.to(audio.device)

    # Split the input audio signals into frames and then flatten to create a tensor
    # of audio frames that can be batch processed.
    frames, timestamps = frame_audio(
        audio,
        frame_size=16000, # to match size of audiogoal (2, 16000)
        hop_size=hop_size,
        sample_rate=AudioCNN.sample_rate,
    ) # frames: (n_sounds, 2 num_channels, num_frames, 16000 frame_size)
    # Remove channel dimension for mono model
    audio_batches, num_channels, num_frames, frame_size = frames.shape

    # convert frames to spectrograms
    spectrograms = compute_spectrogram(
        frames,
        win_length=AudioCNN.win_length,
        hop_length=AudioCNN.hop_length,
        n_fft=AudioCNN.n_fft,
        n_mels=AudioCNN.n_mels,
        window=model.window,
        mel_scale=model.mel_scale,
        downsample=AudioCNN.downsample,
        include_gcc_phat=AudioCNN.include_gcc_phat,
    ) # n_sounds * n_timestamps * 64 * 51 * (channel + nCr(channels, 2))
    spectrograms = spectrograms.flatten(end_dim=1)

    # We're using a DataLoader to help with batching of frames
    dataset = torch.utils.data.TensorDataset(spectrograms)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    # Put the model into eval mode, and not computing gradients while in inference.
    # Iterate over all batches and accumulate the embeddings for each frame.
    model.eval()
    with torch.no_grad():
        embeddings_list = [model(batch) for batch in loader]

    # Concatenate mini-batches back together and unflatten the frames
    # to reconstruct the audio batches
    embeddings = torch.cat(embeddings_list, dim=0)
    embeddings = embeddings.unflatten(0, (audio_batches, num_frames))

    return embeddings, timestamps


def get_scene_embeddings(audio: Tensor, model: torch.nn.Module, *args, **kwargs) -> Tensor:
    """
    This function returns a single embedding for each audio clip.

    Args:
        audio: n_sounds x n_channels x n_samples of audio in the range [-1, 1]. All sounds in
            a batch will be padded/trimmed to the same length.
        model: Loaded model.

    Returns:
        - embeddings, A float32 Tensor with shape
            (n_sounds, model.scene_embedding_size).
    """
    num_sounds, _num_channels, num_samples = audio.shape
    if _num_channels != model.num_channels:
        raise ValueError(
            f"audio input tensor must be have {model._n_input_audio} channels, "
            f"but got {_num_channels}"
        )

    # stack bins and channels
    # audio = audio.reshape(num_sounds * _num_channels, 1, num_samples)

    # multi-channel input to multi-channel model
    embeddings, _ = get_timestamp_embeddings(audio, model, hop_size=SCENE_HOP_SIZE)
    # averaging over frames
    # sounds * frames * features
    embeddings = torch.mean(embeddings, dim=1)

    # Reshape so embeddings for each channel are stacked
    # _, embedding_size = embeddings.shape
    # embeddings = embeddings.reshape(num_sounds, model.num_channels * embedding_size)

    return embeddings