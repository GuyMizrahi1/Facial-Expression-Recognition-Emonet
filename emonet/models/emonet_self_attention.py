import torch
import torch.nn as nn
import torch.nn.functional as F  # Import for using functions like softmax
from emonet.models.emonet import EmoNet


class EmonetWithSelfAttention(nn.Module):
    """
    This class defines a facial expression recognition model that utilizes the EmoNet model
    as a feature extractor and incorporates a self-attention mechanism.
    """

    def __init__(self, emonet_classes=5, emonet_grad=False):
        """
        Initializes the model.

        Args:
            emonet_classes (int): The number of classes the original EmoNet was trained on (5 or 8). Default: 5.
            emonet_grad (bool): Whether to update the pre-trained EmoNet weights during training. Default: False.
        """
        super(EmonetWithSelfAttention, self).__init__()  # Initialize the parent nn.Module class

        # Determine the device (GPU if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ----------------------- #
        # Load Pre-trained EmoNet #
        # ----------------------- #

        # Set the path to the pre-trained EmoNet weights based on the number of classes
        if emonet_classes == 5:
            pre_trained_path = '/content/Facial-Expression-Recognition-Emonet/pretrained/emonet_5.pth'
        else:
            pre_trained_path = '/content/Facial-Expression-Recognition-Emonet/pretrained/emonet_8.pth'

        # Load the pre-trained weights (state_dict) to the appropriate device
        state_dict = torch.load(pre_trained_path, map_location=self.device)

        # Create an instance of the EmoNet model
        self.emonet = EmoNet(n_expression=emonet_classes)

        # Load the pre-trained weights into the EmoNet instance
        self.emonet.load_state_dict(state_dict)

        # Move the EmoNet model to the selected device (GPU or CPU)
        self.emonet = self.emonet.to(self.device)

        # Set whether to update the pre-trained EmoNet weights during training
        for param in self.emonet.parameters():
            param.requires_grad = emonet_grad

        # If not updating pre-trained weights, set the EmoNet model to evaluation mode
        if not emonet_grad:
            self.emonet.eval()

        # ---------------------- #
        # Self-Attention Module #
        # ---------------------- #

        # Create a self-attention module, assuming the EmoNet output has 256 channels
        self.attention = SelfAttention(input_dim=256)

        # -------------------- #
        # Classification Head #
        # -------------------- #

        # Create a fully connected layer to classify the 7 FER classes
        self.fc = nn.Linear(256, 7)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The output tensor containing class scores.
        """
        x = x.to(self.device)  # Move the input tensor to the selected device

        # Extract features from the input image using the pre-trained EmoNet model
        emonet_features = self.emonet.forward(x)

        # -------------------- #
        # Apply Self-Attention #
        # -------------------- #

        # Pass the EmoNet features through the self-attention module
        attended_features = self.attention(emonet_features)

        # ------------------------- #
        # Global Average Pooling #
        # ------------------------- #

        # Average the self-attended features over the spatial dimensions (height and width)
        x = torch.mean(attended_features, dim=(2, 3))

        # ---------------- #
        # Classification #
        # ---------------- #

        # Pass the averaged features through the fully connected layer to get class scores
        x = self.fc(x)
        return x


class SelfAttention(nn.Module):
    """
    Implements a dot-product self-attention mechanism.
    """

    def _init_(self, input_dim):
        """
        Initializes the self-attention module.

        Args:
            input_dim (int): The number of channels in the input features.
        """
        super(SelfAttention, self)._init_()  # Initialize the parent nn.Module class

        # Create convolutional layers to project the input into query, key, and value matrices
        self.query_layer = nn.Conv2d(input_dim, input_dim // 8, kernel_size=1)  # Reduce dimensionality for queries
        self.key_layer = nn.Conv2d(input_dim, input_dim // 8, kernel_size=1)  # Reduce dimensionality for keys
        self.value_layer = nn.Conv2d(input_dim, input_dim, kernel_size=1)  # Keep same dimensionality for values
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling parameter for attention output

    def forward(self, x):
        """
        Defines the forward pass of the self-attention module.

        Args:
            x (torch.Tensor): The input feature tensor.

        Returns:
            torch.Tensor: The output tensor with self-attention applied.
        """
        batch_size, channels, height, width = x.size()  # Get the input dimensions

        # Project the input into query, key, and value matrices
        queries = self.query_layer(x).view(batch_size, -1, height * width).permute(0, 2,
                                                                                   1)  # Reshape and transpose for queries
        keys = self.key_layer(x).view(batch_size, -1, height * width)  # Reshape for keys
        values = self.value_layer(x).view(batch_size, -1, height * width)  # Reshape for values

        # Calculate attention scores using the dot product between queries and keys
        attention_scores = torch.bmm(queries, keys)  # Batch matrix multiplication

        # Apply softmax to the attention scores to get attention weights (normalized to sum to 1)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply the attention weights to the values using matrix multiplication
        attended_features = torch.bmm(values, attention_weights.permute(0, 2, 1))

        # Reshape the attended features back to the original spatial dimensions
        attended_features = attended_features.view(batch_size, channels, height, width)

        # Scale the attended features with the learnable gamma parameter and add them to the original features
        out = self.gamma * attended_features + x

        return out
