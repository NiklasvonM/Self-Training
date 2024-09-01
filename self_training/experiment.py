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
    test_dataset: datasets.MNIST
    confidence_threshold: float
    current_iteration: int
    metrics: list[MetricCollection]
    device: torch.device

    def __init__(self, initial_subset_size: int = 1000, confidence_threshold: float = 0.99) -> None:
        torch.manual_seed(0)
        self.full_train_dataset, self.test_dataset = load_train_test_data()
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

    def run_iteration2(self) -> None:
        self.current_iteration += 1
        train_loader = DataLoader(self.train_subset, batch_size=64, shuffle=True)
        model = train_model(train_loader, device=self.device)
        self._evaluate_iteration(model)
        # Update the training set

    def run_iteration(self) -> bool:
        """
        Run a single iteration of the experiment, saving the new train data as well as evaluation
        metrics on this Experiment instance.
        Return whether or not any new train data was added. This will be false if the model was not
        confident in any more out-of-sample predictions.
        """
        self.current_iteration += 1
        current_number_training_samples = len(self.train_subset_indices)
        train_loader = DataLoader(self.train_subset, batch_size=64, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        model = train_model(train_loader, device=self.device)
        _, accuracy_in_sample = self.predict(model, train_loader)
        predictions_unlabeled, accuracy_unlabeled = self.predict(model, self.get_unlabeled_loader())
        _, accuracy_test = self.predict(model, test_loader)
        self._evaluate_iteration(
            predictions_unlabeled=predictions_unlabeled,
            accuracy_in_sample=accuracy_in_sample,
            accuracy_unlabeled=accuracy_unlabeled,
            accuracy_test=accuracy_test,
        )
        self.update_train_data(predictions_unlabeled)
        new_number_training_samples = len(self.train_subset_indices)
        return new_number_training_samples > current_number_training_samples

    def update_train_data(self, unlabeled_predictions: list[tuple[Digit, float]]) -> None:
        unlabeled_indices = [
            i for i in range(len(self.full_train_dataset)) if i not in self.train_subset_indices
        ]
        unlabeled_subset = Subset(self.full_train_dataset, unlabeled_indices)
        # Add pseudo-labeled data to the training set based on confidence
        new_train_indices: list[int] = []
        new_train_data: list[tuple[torch.Tensor, Digit]] = []
        for i, (pseudo_label, confidence) in enumerate(unlabeled_predictions):
            if confidence > self.confidence_threshold:
                new_train_data.append((unlabeled_subset[i][0], pseudo_label))
                new_train_indices.append(unlabeled_indices[i])  # Add the index
        self.train_subset = torch.utils.data.ConcatDataset([self.train_subset, new_train_data])
        new_train_indices_tensor = torch.tensor(new_train_indices)
        self.train_subset_indices = torch.cat((self.train_subset_indices, new_train_indices_tensor))

    def predict(
        self, model: CNN, data_loader: DataLoader
    ) -> tuple[list[tuple[Digit, float]], float]:
        model.eval()
        predictions: list[tuple[Digit, float]] = []
        labels: list[Digit] = []
        with torch.no_grad():
            for data, target in data_loader:
                output = model(data.to(self.device))
                probabilities = nn.functional.softmax(output, dim=1)
                max_probs, predicted = torch.max(probabilities, 1)
                predictions.extend(zip(predicted.tolist(), max_probs.tolist(), strict=True))
                labels.extend(target.tolist())
        accuracy = accuracy_score([p[0] for p in predictions], labels)
        return predictions, accuracy

    def get_unlabeled_loader(self, batch_size: int = 64) -> DataLoader:
        unlabeled_indices = [
            i for i in range(len(self.full_train_dataset)) if i not in self.train_subset_indices
        ]
        unlabeled_subset = Subset(self.full_train_dataset, unlabeled_indices)
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=batch_size, shuffle=False)
        return unlabeled_loader

    def _evaluate_iteration(
        self,
        predictions_unlabeled: list[tuple[Digit, float]],
        accuracy_in_sample: float,
        accuracy_unlabeled: float,
        accuracy_test: float,
    ) -> None:
        high_confidence_predictions: list[Digit] = []
        high_confidence_true_labels: list[Digit] = []
        low_confidence_predictions: list[Digit] = []
        low_confidence_true_labels: list[Digit] = []
        for i, (_, target) in enumerate(self.get_unlabeled_loader(batch_size=1)):
            true_label: Digit = target[0].item()
            prediction, confidence = predictions_unlabeled[i]
            if confidence > self.confidence_threshold:
                high_confidence_predictions.append(prediction)
                high_confidence_true_labels.append(true_label)
            else:
                low_confidence_predictions.append(prediction)
                low_confidence_true_labels.append(true_label)
        share_correct_train_labels = sum(
            [
                self.train_subset[index_train_subset][1]
                == self.full_train_dataset[index_full_subset][1]
                for index_train_subset, index_full_subset in enumerate(self.train_subset_indices)
            ]
        ) / len(self.train_subset_indices)
        high_confidence_accuracy = (
            accuracy_score(high_confidence_true_labels, high_confidence_predictions)
            if high_confidence_predictions
            else None
        )
        low_confidence_accuracy = (
            accuracy_score(low_confidence_true_labels, low_confidence_predictions)
            if low_confidence_predictions
            else None
        )
        metrics = MetricCollection(
            iteration=self.current_iteration,
            number_training_samples=len(self.train_subset_indices),
            high_confidence_count=len(high_confidence_predictions),
            low_confidence_count=len(low_confidence_predictions),
            accuracy_test=accuracy_test,
            high_confidence_accuracy=high_confidence_accuracy,
            low_confidence_accuracy=low_confidence_accuracy,
            share_correct_train_labels=share_correct_train_labels,
            accuracy_in_sample=accuracy_in_sample,
            accuracy_unlabeled=accuracy_unlabeled,
        )
        print(metrics)
        self.metrics.append(metrics)
