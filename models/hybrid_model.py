import torch
import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * (window_size // 2), 128)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return x

class HybridModel(nn.Module):
    def __init__(self, config):
        super(HybridModel, self).__init__()
        self.cnn = CNNFeatureExtractor(config['input_size'])
        self.lstm = nn.LSTM(128, config['hidden_size'], batch_first=True)
        self.fc = nn.Linear(config['hidden_size'], config['output_size'])

    def forward(self, x):
        # x shape: (batch_size, window_size, num_features)
        x = self.cnn(x)  # Extract features using CNN
        x = x.unsqueeze(1)  # Add sequence dimension for LSTM
        x, _ = self.lstm(x)  # Pass through LSTM
        x = self.fc(x[:, -1, :])  # Get the last output for prediction
        return x

# Function to initialize the model
def initialize_model(config):
    model = HybridModel(config)
    return model
