import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTCNN(nn.Module):
    """
    A standard Convolutional Neural Network (CNN) for MNIST Digit Recognition.
    Includes helper methods to extract intermediate activations for visualization.
    """
    def __init__(self):
        super(MNISTCNN, self).__init__()
        # Conv Layer 1: 1 input channel (grayscale), 32 output channels, kernel size 3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Conv Layer 2: 32 input channels, 64 output channels, kernel size 3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout to prevent overfitting
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        # MNIST input size is 28x28
        # conv1 -> 32 x 28 x 28 -> pool -> 32 x 14 x 14
        # conv2 -> 64 x 14 x 14 -> pool -> 64 x 7 x 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Convolutional layers with pooling and activations
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Flatten for fully connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

    def get_activations(self, x):
        """
        Extract activations of conv1 and conv2 layers for visualization in Streamlit.
        
        Args:
            x (torch.Tensor): Preprocessed input tensor of shape (1, 1, 28, 28)
            
        Returns:
            tuple: (conv1_activations, conv2_activations)
        """
        # Ensure model is in eval mode
        self.eval()
        with torch.no_grad():
            act1 = F.relu(self.conv1(x))
            pooled1 = self.pool(act1)
            act2 = F.relu(self.conv2(pooled1))
        return act1, act2
