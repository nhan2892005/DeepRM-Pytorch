import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import os

class TorchCNNModel(nn.Module):
    """CNN Model."""

    def __init__(self, input_shape, output_shape):
        super(TorchCNNModel, self).__init__()

        self.path_model = '__cache__/model/deeprm.pth'
        self.conv1 = nn.Conv2d(input_shape[0], 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self._compute_flatten_size(input_shape), 256)
        self.fc2 = nn.Linear(256, output_shape)

    def forward(self, x):
        """Forward pass."""
        x = func.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self):
        """Save the model to a file."""
        if not os.path.exists('__cache__/model'):
            os.makedirs('__cache__/model')
        torch.save(self.state_dict(), self.model_path)

    def load(self):
        """Load the model from a file."""
        self.load_state_dict(torch.load(self.model_path))
    
    def _compute_flatten_size(self, input_shape):
        """Compute the size of the flattened features after convolution and pooling."""
        # Create a dummy input to calculate the flattened size
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)  # Batch size 1
            x = func.relu(self.conv1(x))
            x = self.pool(x)
            return x.numel()  # Get total number of elements

# Define input and output shapes
input_shape = (3, 64, 64)  # (channels, height, width)
output_shape = 10  # Number of classes or output size

# Initialize the model
model = TorchCNNModel(input_shape, output_shape)

# Example input
x = torch.randn(8, *input_shape)  # Batch size of 8
output = model(x)

print("Output shape:", output.shape)

# Save the model
model.save()

# Load the model
model.load()
