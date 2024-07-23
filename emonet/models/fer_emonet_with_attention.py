import torch
import torch.nn as nn
import torch.nn.functional as F
from emonet.emonet.models.fer_emonet import FerEmonet


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.size()

        # Flatten x for dot-product attention
        x_flattened = x.view(batch_size, channels, -1)  # shape: (batch_size, channels, height*width)
        x_flattened_transposed = x_flattened.permute(0, 2, 1)  # shape: (batch_size, height*width, channels)

        # Dot product between all pairs of positions in the flattened image
        attention_scores = torch.bmm(x_flattened,
                                     x_flattened_transposed)  # shape: (batch_size, height*width, height*width)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention weights
        # shape: (batch_size, channels, height*width)
        attended = torch.bmm(x_flattened_transposed, attention_weights).permute(0, 2, 1)

        # Reshape to original shape
        attended = attended.view(batch_size, channels, height, width)

        return attended


class FerEmonetWithAttention(nn.Module):
    def __init__(self, emonet_classes=5, emonet_grad=False):
        super(FerEmonetWithAttention, self).__init__()
        self.emonet = FerEmonet(emonet_classes, emonet_grad)  # Assuming FerEmonet is defined elsewhere
        self.self_attention = SelfAttention()
        # Additional layers can be added here if needed

    def forward(self, x):
        emonet_features = self.emonet(x)
        attended_features = self.self_attention(emonet_features)
        # Further processing can be done here before returning the output
        return attended_features
