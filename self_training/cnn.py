import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class CNN(nn.Module):
    def __init__(self, image_dimensions: tuple[int, int] = (28, 28)) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)  # Will be updated dynamically
        self.fc2 = nn.Linear(128, 10)
        # Calculate the correct input size for fc1 dynamically
        dummy_input = torch.randn(1, 1, image_dimensions[0], image_dimensions[1])
        output = self.conv2(self.conv1(dummy_input))
        output = nn.functional.max_pool2d(output, 2)
        fc1_input_size = output.view(1, -1).size(1)
        # Update fc1 with the correct input size
        self.fc1 = nn.Linear(fc1_input_size, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten

        # Dynamically calculate the size of the flattened tensor
        fc1_input_size = x.size(1)

        # Recreate fc1 with the correct input size if needed
        if fc1_input_size != self.fc1.in_features:
            self.fc1 = nn.Linear(fc1_input_size, 128)

        x = torch.nn.functional.linear(x, self.fc1.weight, self.fc1.bias)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


def train_model(train_loader: DataLoader, train_epochs=10) -> CNN:
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for _ in tqdm(range(train_epochs)):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
    return model
