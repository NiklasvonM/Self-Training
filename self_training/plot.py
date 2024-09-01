from pathlib import Path

import matplotlib.pyplot as plt

from .experiment_result import ExperimentResult, load_all_experiment_results, load_experiment_result


def plot_experiment_result(experiment_result: ExperimentResult | Path) -> None:
    if isinstance(experiment_result, Path):
        experiment_result = load_experiment_result(experiment_result)
    metrics = experiment_result.metrics
    iterations = [metric.iteration for metric in metrics]
    test_accuracy = [metric.test_accuracy for metric in metrics]

    plt.plot(iterations, test_accuracy)
    plt.xlabel("Iteration")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs. Iteration")
    plt.show()


def plot_all_experiment_results(folder: str | Path = "output") -> None:
    experiment_results = load_all_experiment_results(folder)
    plot_multiple_experiment_results(experiment_results)


def plot_multiple_experiment_results(experiment_results: list[ExperimentResult]) -> None:
    experiment_results.sort(key=lambda experiment_result: experiment_result.confidence_threshold)
    confidence_thresholds = [
        experiment_result.confidence_threshold for experiment_result in experiment_results
    ]
    accuracy_improvements: list[float] = []
    for experiment_result in experiment_results:
        accuracy_improvements.append(
            experiment_result.metrics[-1].test_accuracy - experiment_result.metrics[0].test_accuracy
        )

    plt.scatter(confidence_thresholds, accuracy_improvements)
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Accuracy Improvements")
    plt.title("Accuracy Improvements vs. Confidence Threshold")
    plt.show()
