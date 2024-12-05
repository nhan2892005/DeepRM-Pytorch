import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import os

class TorchCNNModel(nn.Module):
    """CNN Model."""

    def __init__(self, input_shape, output_shape):
        super(TorchCNNModel, self).__init__()
        print(input_shape)
        self.model_path = '__cache__/model/deeprm.pth'
        self.conv1 = nn.Conv2d(kernel_size=(3, 3), stride=1, padding='same', out_channels=16, in_channels=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self._compute_flatten_size(input_shape), 256)
        self.fc2 = nn.Linear(256, output_shape)

    def forward(self, x):
        """Forward pass."""
        print(x.shape)
        x = func.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
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
    # Create a dummy input to calculate the flattened size
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)  # Batch size 1
            x = func.relu(self.conv1(x))  # Apply convolution
            x = self.pool(x)  # Apply pooling
            x = self.dropout(x)  # Apply dropout
            x = self.flatten(x)  # Flatten the output
            return x.shape[1]  # Return the number of features after flattening

# Define input and output shapes
input_shape = (10, 400)  # (height, width, channels)
output_shape = 31  # Number of classes or output size

# Initialize the model
model = TorchCNNModel(input_shape, output_shape)

# Example input
x = torch.randn(1, *input_shape)  # Batch size of 8
output = model(x)

print("Output shape:", output.shape)

# Save the model
model.save()

# Load the model
model.load()
