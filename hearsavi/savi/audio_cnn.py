#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)

class AudioCNN(nn.Module):
    r"""A Simple 3-Conv CNN followed by a fully connected layer
    Takes in observations of audio-sensor and produces an embedding of the spectrograms.
    Args:
        output_size: The size of the embedding vector
    """
    sample_rate = 16000
    num_channels = 2
    embedding_size = 128
    scene_embedding_size = embedding_size
    timestamp_embedding_size = embedding_size

    def __init__(self, cnn_dims=(65, 26), num_input_channels=None):
        super().__init__()
        self._n_input_audio = (num_input_channels or self.num_channels) # input channel 'spectrogram' (65, 26, 2)

        cnn_dims = np.array(cnn_dims, dtype=np.float32) # single channel spectrogram shape


        if cnn_dims[0] < 30 or cnn_dims[1] < 30:
            self._cnn_layers_kernel_size = [(5, 5), (3, 3), (3, 3)]
            self._cnn_layers_stride = [(2, 2), (2, 2), (1, 1)]
        else:
            self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
            self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self._n_input_audio,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            # nn.ReLU(True),
            # nn.Conv2d(
            #     in_channels=64,
            #     out_channels=32,
            #     kernel_size=self._cnn_layers_kernel_size[3],
            #     stride=self._cnn_layers_stride[3],
            # ),
            #  nn.ReLU(True),
            Flatten(),
            nn.Linear(64 * cnn_dims[0] * cnn_dims[1], self.embedding_size),
            nn.ReLU(True),
        )

        self.layer_init()

    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.

        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def layer_init(self):
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, audio_observations):
        # input shape [BATCH x CHANNEL x HEIGHT X WIDTH] (batch, 2, 65, 26)
        # permute tensor to dimension 
        return self.cnn(audio_observations)