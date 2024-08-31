import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)  # Will be updated dynamically
        self.fc2 = nn.Linear(128, 10)

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


# See https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
mnist_mean = 0.1307
mnist_std = 0.3081
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((mnist_mean,), (mnist_std,))]
)

train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

initial_subset_size = 1000
train_subset_indices = torch.randperm(len(train_dataset))[:initial_subset_size]
train_subset = Subset(train_dataset, train_subset_indices)

model = CNN()

# Calculate the correct input size for fc1 dynamically
dummy_input = torch.randn(1, 1, 28, 28)  # MNIST images are 28x28
output = model.conv2(model.conv1(dummy_input))
output = nn.functional.max_pool2d(output, 2)
fc1_input_size = output.view(1, -1).size(1)

# Update fc1 with the correct input size
model.fc1 = nn.Linear(fc1_input_size, 128)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Pseudo-labeling iterations
num_iterations = 10
confidence_threshold = 0.99
out_of_sample_accuracies: list[float] = []
train_epochs = 10

for iteration in range(num_iterations):
    current_train_size = len(train_subset)
    print(f"Iteration {iteration + 1}: Current training size = {current_train_size}")
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    model.train()
    for _ in tqdm(range(train_epochs)):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Predict on the remaining training data (unlabeled)
    unlabeled_indices = [i for i in range(len(train_dataset)) if i not in train_subset_indices]
    unlabeled_subset = Subset(train_dataset, unlabeled_indices)
    unlabeled_loader = DataLoader(unlabeled_subset, batch_size=64, shuffle=False)

    model.eval()
    pseudo_labels = []
    high_confidence_count = 0
    high_confidence_predictions = []
    high_confidence_true_labels = []
    low_confidence_predictions = []
    low_confidence_true_labels = []
    with torch.no_grad():
        for data, target in unlabeled_loader:
            output = model(data)
            probabilities = nn.functional.softmax(output, dim=1)
            max_probs, predicted = torch.max(probabilities, 1)
            pseudo_labels.extend(zip(predicted.tolist(), max_probs.tolist(), strict=True))
            high_confidence_count += (max_probs > confidence_threshold).sum().item()

            for i in range(len(predicted)):
                if max_probs[i] > confidence_threshold:
                    high_confidence_predictions.append(predicted[i].item())
                    high_confidence_true_labels.append(target[i].item())
                else:
                    low_confidence_predictions.append(predicted[i].item())
                    low_confidence_true_labels.append(target[i].item())

    if high_confidence_predictions:
        high_confidence_accuracy = accuracy_score(
            high_confidence_true_labels, high_confidence_predictions
        )
        print(
            f"Iteration {iteration + 1}: High-confidence accuracy = {high_confidence_accuracy} "
            f"(n={len(high_confidence_predictions)})"
        )
    else:
        print(f"Iteration {iteration + 1}: No high-confidence predictions made.")

    if low_confidence_predictions:
        low_confidence_accuracy = accuracy_score(
            low_confidence_true_labels, low_confidence_predictions
        )
        print(
            f"Iteration {iteration + 1}: Low-confidence accuracy = {low_confidence_accuracy} "
            f"(n={len(low_confidence_predictions)})"
        )
    else:
        print(f"Iteration {iteration + 1}: No low-confidence predictions made.")

    # Add pseudo-labeled data to the training set based on confidence
    new_train_indices = []
    new_train_data = []
    for i, (pseudo_label, confidence) in enumerate(pseudo_labels):
        if confidence > confidence_threshold:
            new_train_data.append((unlabeled_subset[i][0], pseudo_label))
            new_train_indices.append(unlabeled_indices[i])  # Add the index
    train_subset = torch.utils.data.ConcatDataset([train_subset, new_train_data])
    new_train_indices_tensor = torch.tensor(new_train_indices)
    train_subset_indices = torch.cat((train_subset_indices, new_train_indices_tensor))
    # Evaluate on the test set
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model.eval()
    test_predictions = []
    test_true_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            test_predictions.extend(predicted.tolist())
            test_true_labels.extend(target.tolist())

    accuracy = accuracy_score(test_true_labels, test_predictions)
    out_of_sample_accuracies.append(accuracy)
    print(f"Iteration {iteration + 1}: Out-of-sample accuracy = {accuracy}")
