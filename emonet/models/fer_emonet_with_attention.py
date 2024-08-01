import torch
import torch.nn as nn
import torch.nn.functional as F
from emonet.models.emonet import EmoNet


# class SelfAttention(nn.Module):
#     def __init__(self):
#         super(SelfAttention, self).__init__()
#
#     def forward(self, x):
#         # x shape: (batch_size, channels, height, width)
#         batch_size, channels, height, width = x.size()
#
#         # Flatten x for dot-product attention
#         x_flattened = x.view(batch_size, channels, -1)  # shape: (batch_size, channels, height*width)
#         x_flattened_transposed = x_flattened.permute(0, 2, 1)  # shape: (batch_size, height*width, channels)
#
#         # Dot product between all pairs of positions in the flattened image
#         attention_scores = torch.bmm(x_flattened,
#                                      x_flattened_transposed)  # shape: (batch_size, height*width, height*width)
#         attention_weights = F.softmax(attention_scores, dim=-1)
#
#         # Apply attention weights
#         # shape: (batch_size, channels, height*width)
#         attended = torch.bmm(x_flattened_transposed, attention_weights).permute(0, 2, 1)
#
#         # Reshape to original shape
#         attended = attended.view(batch_size, channels, height, width)
#
#         return attended

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # Compute attention scores
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention weights
        attended = torch.bmm(attention_weights, values)
        return attended


class FerEmonetWithAttention(nn.Module):
    def __init__(self, emonet_classes=5):
        super(FerEmonetWithAttention, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pre-trained EmoNet model
        self.emonet = EmoNet(n_expression=emonet_classes)
        pre_trained_path = '/content/Facial-Expression-Recognition-Emonet/pretrained/emonet_5.pth' if (
                emonet_classes == 5) else '/content/Facial-Expression-Recognition-Emonet/pretrained/emonet_8.pth'
        state_dict = torch.load(pre_trained_path, map_location=self.device)
        self.emonet.load_state_dict(state_dict)
        self.emonet = self.emonet.to(self.device)
        self.emonet.eval()

        # Self-Attention mechanism
        self.self_attention = SelfAttention(embed_dim=256)

        # Assuming that is the chosen final layer configuration
        self.conv1x1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.batch_norm = nn.BatchNorm2d(128)
        self.mean_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 7)

    def forward(self, x):
        x = x.to(self.device)
        emonet_features = self.emonet.forward(x)

        # Apply self-attention
        emonet_features = emonet_features.unsqueeze(1)
        attended_features = self.self_attention(emonet_features)
        attended_features = attended_features.squeeze(1)

        # Final layers
        x = self.conv1x1(attended_features)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.mean_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
