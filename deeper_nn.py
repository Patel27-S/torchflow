import torch
import torch.nn as nn

# Define a deeper neural network
class DeeperNN(nn.Module):
    def __init__(self):
        super(DeeperNN, self).__init__()
        self.fc1 = nn.Linear(2, 8)  # Input layer to 8 neurons
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 6)  # Hidden layer 1
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(6, 4)  # Hidden layer 2
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(4, 1)  # Output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

# Create the model
model = DeeperNN()
print(model)
