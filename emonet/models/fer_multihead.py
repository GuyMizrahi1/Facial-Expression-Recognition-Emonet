import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .emonet import EmoNet


# New Model for Assignment
class FerMultihead(nn.Module):
    def __init__(self, emonet_classes=8, emonet_grad=False, embed_dim=64, num_heads=2, patch_size=16):
        super(FerMultihead, self).__init__()
        """
        params:
        emonet_classes: The number of classes the original emonet was trained on.
                        There are 2 pre-trained models: 5 and 8 classes. default=5
        emonet_grad: chooses if during training, updating emonet pre-trained weights or not. default=False.
        embed_dim: The length of each patch vector, after applying linear projection.
        num_heads: number of attention heads
        patch_size: example, 16X16 
        """

        # Specify the device - cuda / cpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # ------------ #
        # Emonet Block #
        # ------------ #

        # pretrained 5 or 8
        if emonet_classes == 5:
            pre_trained_path = "C:/Users/sdole/PycharmProjects/emonet_Project/pretrained/emonet_5.pth"
        else:
            pre_trained_path = "C:/Users/sdole/PycharmProjects/emonet_Project/pretrained/emonet_8.pth"
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

        # ---------------------------------- #
        # CNN Block and Multi-head Attention #
        # ---------------------------------- #

        # cnn block
        self.cnn_block = cnnBlock()
        # multi-head attention
        self.multi_head = multi_head_module(embed_dim=embed_dim, num_heads=num_heads, patch_size=patch_size)

        # ----------- #
        # Final layer #
        # ----------- #

        # flatten
        self.flatten = nn.Flatten()
        # reduce from 16*64 to 64
        self.fc1 = nn.Linear(in_features=16*embed_dim, out_features=64)
        self.bn_fc1 = nn.BatchNorm1d(num_features=64)
        self.final_layer = nn.Linear(in_features=64, out_features=7)

    def forward(self, x):
        x = x.to(self.device)
        # emonet's forward function
        emonet_features = self.emonet.forward(x) # out (batch_size,256,64,64)
        # cnn block
        x = self.cnn_block(emonet_features) # out (batch_size,16,64,64)
        # multi head block
        x = self.multi_head(x) # out (batch_size,16,64)
        # flatten and final fc layers
        x = self.flatten(x) # out (batch_size, 16*64)
        x = self.fc1(x) # out (batch_size, 64)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.final_layer(x)
        return x


class cnnBlock(nn.Module):
    def __init__(self):
        super(cnnBlock, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        return x


class multi_head_module(nn.Module):
    def __init__(self, embed_dim, num_heads, patch_size):
        super(multi_head_module, self).__init__()
        '''
        params: 
            embed_dim: The length of each patch vector, after applying linear projection. 
                        Using linear projection in order to make vectors shorter. 
            num_heads: The number of attention heads.
            patch_size: The patch size. example: 16, makes as 16X16 patches. 
        '''
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size

        # First flatten the patch into a vector, and then use linear projection
        patch_length = self.patch_size * self.patch_size * 16
        self.projection = nn.Linear(in_features=patch_length, out_features=self.embed_dim)

        # Positional encoding of patches - in oder to preserve spatial location
        # (1,num_patches, patch_length=embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, (64 // self.patch_size) ** 2, self.embed_dim))

        # Use a multi head module (pytorch)
        self.multi_head_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)

        # Splitting into patches
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Extracts the values from the tensor.shape
        batch_size, feature_maps, height, width = x.shape

        # Unfold the input into patches. Input: (batch size, 256,64,64) -> (32, 256, 64, 64)
        # Output: (batch size, patch_length, num_patches) -> (32,256*16*16, 16)
        x_patches = self.unfold(x)  # patch_length = 256*patch_size^2 = 256*16*16

        # Transpose and reshape to (batch_size, num_patches, patch_length)
        x_patches = x_patches.transpose(1, 2).reshape(batch_size, -1, self.patch_size * self.patch_size * feature_maps)

        # Apply linear projection to each patch
        x_proj = self.projection(x_patches)  # Result Shape: (batch_size, num_patches, embed_dim)

        # Add positional encoding
        x_proj = x_proj + self.positional_encoding

        # Prepare for multi-head attention
        x_proj = x_proj.permute(1, 0, 2)  # Shape: (num_patches, batch_size, embed_dim)

        # Apply multi-head attention - the second argument are attn_weigths, I can save them later for interpretability
        attn_output, _ = self.multi_head_attn(x_proj, x_proj, x_proj)  # Shape: (num_patches, batch_size, embed_dim)

        # Transpose back to (batch_size, num_patches, embed_dim)
        attn_output = attn_output.permute(1, 0, 2)


        return attn_output
