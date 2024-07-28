import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from emonet.models.emonet import EmoNet


# New Model for Assignment
class FerEmonet(nn.Module):
    def __init__(self, emonet_classes=5, emonet_grad=False, final_layer_type=1):
        super(FerEmonet, self).__init__()
        """
        params:
        emonet_classes: The number of classes the original emonet_o was trained on.
                        There are 2 pre-trained models: 5 and 8 classes. default=5
        emonet_grad: chooses if during training, updating emonet_o pre-trained weights or not. default=False.
        final_layers_type: The type of final layers. default=1. can get: 1, 2 or 3
        """
        # Specify the device - cuda / cpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # ------------ #
        # Emonet Block #
        # ------------ #

        # pretrained 5 or 8
        if emonet_classes == 5:
            pre_trained_path = '/content/Facial-Expression-Recognition-Emonet/pretrained/emonet_5.pth'
        else:
            pre_trained_path = '/content/Facial-Expression-Recognition-Emonet/pretrained/emonet_8.pth'
        # loading
        state_dict = torch.load(pre_trained_path, map_location=self.device)
        self.emonet = EmoNet(n_expression=emonet_classes)
        self.emonet.load_state_dict(state_dict)
        # ensures that the pre-trained model is on GPU - all subsequent operations (training etc.) on gpu
        self.emonet = self.emonet.to(self.device)

        # whether to update pre-trained weights or not
        for param in self.emonet.parameters():
            param.requires_grad = emonet_grad

        # setting model to evaluation modde if we want
        if not emonet_grad:
            self.emonet.eval()
        # ----------- #
        # Final layer #
        # ----------- #
        self.final_layer_type = final_layer_type
        # final layers config
        if final_layer_type == 1:
            # single conv layer and avg pooling filter 1X1
            # converting multiple channels into one channel per image
            self.conv1x1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
            self.batch_norm = nn.BatchNorm2d(128)
            # average pooling - max of each channel -> (batch size, 128)
            self.mean_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(128, 7)  # 7 classes for FER
        # single conv and flattening
        elif final_layer_type == 2:
            # converting multiple channels into one channel per image
            self.conv1x1 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
            self.batch_norm = nn.BatchNorm2d(1)
            # flatten the channel
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(64 * 64, 7)  # 7 classes for FER
        # two conv layers and acg pooling filter 2X2
        elif final_layer_type == 3:
            # Model with two conv layers, average pooling, and flattening
            self.conv1x1a = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
            self.batch_norm1 = nn.BatchNorm2d(128)
            self.conv1x1b = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
            self.batch_norm2 = nn.BatchNorm2d(1)
            self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(32 * 32, 7)

    def forward(self, x):
        # activating emonet's forward function
        x = x.to(self.device)
        emonet_features = self.emonet.forward(x)

        if self.final_layer_type == 1:
            x = self.conv1x1(emonet_features)
            x = self.batch_norm(x)
            x = F.relu(x)
            x = self.mean_pool(x)
            x = x.view(x.size(0), -1)  # (batch size, 128, 1, 1) -> (batch size, 128)
        elif self.final_layer_type == 2:
            x = self.conv1X1(emonet_features)
            x = self.batch_norm(x)
            x = F.relu(x)
            x = self.flatten(x)
        elif self.final_layer_type == 3:
            x = self.conv1x1a(emonet_features)
            x = self.batch_norm1(x)
            x = F.relu(x)
            x = self.conv1x1b(x)
            x = self.batch_norm2(x)
            x = F.relu(x)
            x = self.avg_pool(x)
            x = self.flatten(x)
        # classification head
        x = self.fc(x)
        return x
