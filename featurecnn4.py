from typing import List, Union, Tuple

import torch
from nn import Sequential
from torch import Tensor
from torch import nn as nn

from savable import Savable


class FeatureCNN4(nn.Module, Savable):
    def __init__(self, tank_crop: bool):
        nn.Module.__init__(self)
        Savable.__init__("classifier")
        reconstruction_initialized: bool = False
        self._conv_list: nn.ModuleList = nn.ModuleList()
        self._deconv_list: nn.ModuleList = nn.ModuleList()
        self._bias_list: List[Tensor] = []
        self._activation: Sequential = nn.Sequential(
        nn.ReLU(),
        nn.MaxPool2d(2, 2, return_indices=True)
        )
        self._channel_list: List[int] = [1, 16, 32, 64, 128]
        for a, b in zip(self._channel_list, self._channel_list[1:]):
            self._conv_list.append(
            nn.Conv2d(a, b, 3, 1, 1)
            )
            self._deconv_list.append(
            nn.ConvTranspose2d(b, a, 3, 1, 1, bias=False)
            )

        if tank_crop:
            self.fc1 = nn.Linear(2048, 1024)
        else:
            self.fc1 = nn.Linear(8192, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, 256)
            self.fc4 = nn.Linear(256, 128)
            self.fc5 = nn.Linear(128, 10)
            self.relu = nn.ReLU()

            self.relu = nn.ReLU()
            self.softmax = torch.nn.Softmax()

    def forward(self, x) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        features: List[Tensor] = []
        switches: List[Tensor] = []
        features.append(x)
        temp = []
        for conv in self.conv_list:
            temp = self._activation(conv(features[-1]))
        features.append(temp[0])
        switches.append(temp[1])
        features = features[1:]
        classification: Tensor = self._classify(x)
        upscaled_features: List[Tensor] = []
        if self.training:
            return x
        else:
            if self._deconv_list is None:
                self._initialize_deconv_list()
        for i, feature in enumerate(features):
            upscaled_features.append(
                self._deconv(feature, i, switches)
            )
        return classification, upscaled_features

    def train(self, flag: bool):
        if flag:
            self._deconv_list = None
        super().train(flag)

    def _classify(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def _initialize_deconv_list(self):
        self._bias_list = []
        for conv, deconv in zip(self.conv_list, self.deconv_list):
            deconv.weight = nn.Parameter(conv.weight.transpose(2,3))
            self._bias_list.append(conv.bias)

    def _deconv(self, feature_map: Tensor, layer_location: int, switch: List[Tensor]):
        if layer_location == 0:
            return feature_map
        else:
            #todo: add code here to do the deconv stuff to get "out"
            backwards: Sequential = nn.Sequential(
                nn.MaxUnpool2d(2, 2),
                nn.ReLU(),
            )
            out: Tensor = torch.Tensor()
            for deconv in self._deconv_list:
                out = deconv(backwards(feature_map[layer_location], switch[feature_map]))
            return self._deconv(out, layer_location - 1)