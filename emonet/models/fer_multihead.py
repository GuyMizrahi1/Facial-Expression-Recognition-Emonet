import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from emonet.models.emonet import EmoNet


# New Model for Assignment
class FerMultihead(nn.Module):
    def __init__(self, emonet_classes=5, emonet_grad=False, embed_dim=256, num_heads=2, patch_size=16):
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
            pre_trained_path = "../../pretrained/emonet_5.pth"
        else:
            pre_trained_path = "../../pretrained/emonet_8.pth"
        # loading
        state_dict = torch.load(pre_trained_path, map_location=self.device)
        self.emonet = EmoNet(n_expression=emonet_classes)
        self.emonet.load_state_dict(state_dict)
        # ensures that the pre-trained model is on GPU - all subsequent operations (training etc.) on gpu
        self.emonet = self.emonet.to(self.device)

        # whether to update pre-trained weights or not
        for param in self.emonet.parameters():
            param.requires_grad = emonet_grad

        # --------------- #
        # Attention layer #
        # --------------- #

        self.multi_head = multi_head_module(embed_dim=embed_dim, num_heads=num_heads, patch_size=patch_size)

        # ----------- #
        # Final layer #
        # ----------- #

        # converting multiple channels into one channel per image
        self.conv1x1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.batch_norm = nn.BatchNorm2d(128)
        # average pooling - max of each channel -> (batch size, 128)
        self.mean_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 7)

    def forward(self, x):
        # activating emonet's forward function
        x = x.to(self.device)
        emonet_features = self.emonet.forward(x)

        # Apply Multi Head Attention
        attn_out = self.multi_head(emonet_features)

        # need to change the layers afterwords
        x = self.conv1x1(attn_out)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.mean_pool(x)
        x = x.view(x.size(0), -1)  # (batch size, 128, 1, 1) -> (batch size, 128)
        x = self.fc(x)
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
        patch_length = self.patch_size * self.patch_size * 256
        self.projection = nn.Linear(in_features=patch_length, out_features=self.embed_dim)

        # Positional encoding of patches - in oder to preserve spatial location
        # (1,num_patches, patch_length=embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, (64 // self.patch_size) ** 2, self.embed_dim))

        # Use a multi head module from pytorch
        self.multi_head_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)

        # Splitting into patches
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Extracts the values from the tensor.shape
        batch_size, feature_maps, height, width = x.shape

        # Unfold the input into patches. (batch size, 256,64,64) -> (batch size, patch_length, num_patches)
        x_patches = self.unfold(x)  # patch_length = 256*patch_height*patch_length = 256*16*16

        # Transpose and reshape to (batch_size, num_patches, patch_length)
        x_patches = x_patches.transpose(1, 2).reshape(batch_size, -1, self.patch_size * self.patch_size * feature_maps)

        # Apply linear projection to each patch
        x_proj = self.projection(x_patches)  # Result Shape: (batch_size, num_patches, embed_dim)

        # Add positional encoding
        x_proj = x_proj + self.positional_encoding

        # Prepare for multi-head attention
        x_proj = x_proj.permute(1, 0, 2)  # Shape: (num_patches, batch_size, embed_dim)

        # Apply multi-head attention - the secong argument are attn_weigths, I can save them later for interpretability
        attn_output, _ = self.multi_head_attn(x_proj, x_proj, x_proj)  # Shape: (num_patches, batch_size, embed_dim)

        # Transpose back to (batch_size, num_patches, embed_dim)
        attn_output = attn_output.permute(1, 0, 2)

        return attn_output
