import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Input layer -> Hidden layer -> Output layer
        self.fc1 = nn.Linear(2, 4)  # Input: 2 features, Hidden: 4 neurons
        self.relu = nn.ReLU()       # Activation function
        self.fc2 = nn.Linear(4, 1)  # Output: 1 neuron (binary classification)
        self.sigmoid = nn.Sigmoid() # Sigmoid activation for binary output

    def forward(self, x):
        x = self.fc1(x)       # Input -> Hidden
        x = self.relu(x)      # Activation
        x = self.fc2(x)       # Hidden -> Output
        x = self.sigmoid(x)   # Sigmoid activation
        return x

# Create the model
model = SimpleNN()
print(model)
