from typing import Literal

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .cnn import CNN, train_model
from .experiment_result import ExperimentResult
from .metric_collection import MetricCollection

Digit = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def load_train_test_data() -> tuple[datasets.MNIST, datasets.MNIST]:
    # See https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    return train_dataset, test_dataset


class Experiment:
    full_train_dataset: datasets.MNIST
    train_subset_indices: torch.Tensor
    train_subset: Dataset
    test_loader: DataLoader
    confidence_threshold: float
    current_iteration: int
    metrics: list[MetricCollection]
    device: torch.device

    def __init__(self, initial_subset_size: int = 1000, confidence_threshold: float = 0.99) -> None:
        torch.manual_seed(0)
        self.full_train_dataset, test_dataset = load_train_test_data()
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        self.train_subset_indices = torch.randperm(len(self.full_train_dataset))[
            :initial_subset_size
        ]
        self.train_subset = Subset(self.full_train_dataset, self.train_subset_indices)
        self.confidence_threshold = confidence_threshold
        self.current_iteration = 0
        self.metrics = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self, number_iterations: int = 10) -> None:
        """
        Run the experiment for a specified number of iterations.
        """
        for _ in range(number_iterations):
            should_continue = self.run_iteration()
            if not should_continue:
                break

    def get_result(self) -> ExperimentResult:
        return ExperimentResult(
            confidence_threshold=self.confidence_threshold,
            metrics=self.metrics,
        )

    def run_iteration(self) -> bool:
        """
        Run a single iteration of the experiment, saving the new train data as well as evaluation
        metrics on this Experiment instance.
        Return whether or not any new train data was added. This will be false if the model was not
        confident in any more out-of-sample predictions.
        """
        self.current_iteration += 1
        train_loader = DataLoader(self.train_subset, batch_size=64, shuffle=True)
        model = train_model(train_loader, device=self.device)
        self._evaluate_iteration(model)
        # Predict on the remaining training data (unlabeled)
        unlabeled_indices = [
            i for i in range(len(self.full_train_dataset)) if i not in self.train_subset_indices
        ]
        unlabeled_subset = Subset(self.full_train_dataset, unlabeled_indices)
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=64, shuffle=False)

        model.eval()
        pseudo_labels = self._get_pseudo_labels(model, unlabeled_loader)
        # Add pseudo-labeled data to the training set based on confidence
        new_train_indices: list[int] = []
        new_train_data: list[tuple[torch.Tensor, Digit]] = []
        for i, (pseudo_label, confidence) in enumerate(pseudo_labels):
            if confidence > self.confidence_threshold:
                new_train_data.append((unlabeled_subset[i][0], pseudo_label))
                new_train_indices.append(unlabeled_indices[i])  # Add the index
        self.train_subset = torch.utils.data.ConcatDataset([self.train_subset, new_train_data])
        new_train_indices_tensor = torch.tensor(new_train_indices)
        self.train_subset_indices = torch.cat((self.train_subset_indices, new_train_indices_tensor))
        return len(new_train_indices) > 0

    def _get_pseudo_labels(self, model: CNN, data_loader: DataLoader) -> list[tuple[Digit, float]]:
        model.eval()
        pseudo_labels: list[tuple[Digit, float]] = []
        with torch.no_grad():
            for data, _ in data_loader:
                output = model(data.to(self.device))
                probabilities = nn.functional.softmax(output, dim=1)
                max_probs, predicted = torch.max(probabilities, 1)
                pseudo_labels.extend(zip(predicted.tolist(), max_probs.tolist(), strict=True))
        return pseudo_labels

    def _get_unlabeled_loader(self) -> DataLoader:
        unlabeled_indices = [
            i for i in range(len(self.full_train_dataset)) if i not in self.train_subset_indices
        ]
        unlabeled_subset = Subset(self.full_train_dataset, unlabeled_indices)
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=64, shuffle=False)
        return unlabeled_loader

    def _evaluate_iteration(self, model: CNN) -> None:
        model.eval()
        high_confidence_count: int = 0
        high_confidence_predictions: list[Digit] = []
        high_confidence_true_labels: list[Digit] = []
        low_confidence_predictions: list[Digit] = []
        low_confidence_true_labels: list[Digit] = []
        unlabeled_loader = self._get_unlabeled_loader()
        with torch.no_grad():
            for data, target in unlabeled_loader:
                output = model(data.to(self.device))
                target = target.to(self.device)
                probabilities = nn.functional.softmax(output, dim=1)
                max_probs, predicted = torch.max(probabilities, 1)
                high_confidence_count += (max_probs > self.confidence_threshold).sum().item()
                for i in range(len(predicted)):
                    if max_probs[i] > self.confidence_threshold:
                        high_confidence_predictions.append(predicted[i].item())
                        high_confidence_true_labels.append(target[i].item())
                    else:
                        low_confidence_predictions.append(predicted[i].item())
                        low_confidence_true_labels.append(target[i].item())
        test_predictions: list[Digit] = []
        test_true_labels: list[Digit] = []
        with torch.no_grad():
            for data, target in self.test_loader:
                output = model(data.to(self.device))
                target = target.to(self.device)
                _, predicted = torch.max(output.data, 1)
                test_predictions.extend(predicted.tolist())
                test_true_labels.extend(target.tolist())
        share_correct_train_labels = sum(
            [
                self.train_subset[index_train_subset][1]
                == self.full_train_dataset[index_full_subset][1]
                for index_train_subset, index_full_subset in enumerate(self.train_subset_indices)
            ]
        ) / len(self.train_subset_indices)
        metrics = MetricCollection(
            iteration=self.current_iteration,
            number_training_samples=len(self.train_subset_indices),
            high_confidence_accuracy=accuracy_score(
                high_confidence_true_labels, high_confidence_predictions
            )
            if high_confidence_predictions
            else None,
            high_confidence_count=high_confidence_count,
            low_confidence_accuracy=accuracy_score(
                low_confidence_true_labels, low_confidence_predictions
            )
            if low_confidence_predictions
            else None,
            low_confidence_count=len(low_confidence_predictions),
            test_accuracy=accuracy_score(test_true_labels, test_predictions),
            share_correct_train_labels=share_correct_train_labels,
        )
        print(metrics)
        self.metrics.append(metrics)
