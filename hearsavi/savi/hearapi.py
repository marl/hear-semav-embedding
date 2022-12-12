"""
SAVI audio subnetwork model for HEAR.

Adapted from
https://github.com/marl/sound-spaces/tree/main/ss_baselines/savi
"""

from collections import OrderedDict
from typing import Tuple

import torch
import torch.fft
import librosa
import numpy as np
from torch import Tensor
from skimage.measure import block_reduce

from hearsavi.util import frame_audio
from hearsavi.savi.audio_cnn import AudioCNN

# Default hop_size in milliseconds
TIMESTAMP_HOP_SIZE = 50
SCENE_HOP_SIZE = 250
n_fft = 512

# Number of frames to batch process for timestamp embeddings
BATCH_SIZE = 512


def compute_spectrogram(audiogoal):
    def compute_stft(signal):
        n_fft = 512
        hop_length = 160
        win_length = 400
        stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
        stft = block_reduce(stft, block_size=(1, 4, 4), func=np.mean) # downsampling
        return stft

    channel1_magnitude = np.log1p(compute_stft(audiogoal[:,0])) # n_sounds * first channel samples
    channel2_magnitude = np.log1p(compute_stft(audiogoal[:,1]))
    spectrogram = np.stack([channel1_magnitude, channel2_magnitude], axis=-1)

    return spectrogram


def load_model(model_file_path: str = "", net_prefix: str = 'actor_critic.net.goal_encoder') -> torch.nn.Module:
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
        if not net_prefix.endswith('.'):
            net_prefix +=  '.'
        audio_model.load_state_dict(
            {
                k.replace(net_prefix, ''): v
                for k, v in loaded_model['state_dict'].items()
                if k.startswith(net_prefix)
            }
        )
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
    frames = frames.squeeze(dim=1) # ignore for binaural
    audio_batches, num_channels, num_frames, frame_size = frames.shape
    # frames = frames.flatten(end_dim=1)
    # stack first two dimensions so for mono it's (sounds*frames, frame size)
    frames = frames.reshape(-1, frames.shape[1], frames.shape[3]) # reshape to stack binaural frames

    # convert frames to spectrograms
    spectrograms = Tensor(compute_spectrogram(frames.cpu().detach().numpy())) # size * 65 * 26 * channel


    # We're using a DataLoader to help with batching of frames
    dataset = torch.utils.data.TensorDataset(spectrograms)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    # Put the model into eval mode, and not computing gradients while in inference.
    # Iterate over all batches and accumulate the embeddings for each frame.
    model.eval()
    with torch.no_grad():
        embeddings_list = [model(batch[0]) for batch in loader]

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
