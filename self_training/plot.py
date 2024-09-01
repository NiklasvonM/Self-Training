from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .experiment_result import ExperimentResult, load_all_experiment_results, load_experiment_result


def plot_experiment_result(experiment_result: ExperimentResult | Path) -> None:
    if isinstance(experiment_result, Path):
        experiment_result = load_experiment_result(experiment_result)
    metrics = experiment_result.metrics
    iterations = [metric.iteration for metric in metrics]
    accuracy_test = [100 * metric.accuracy_test for metric in metrics]

    plt.plot(iterations, accuracy_test)
    plt.xlabel("Iteration")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy vs. Iteration")
    plt.show()


def plot_all_experiment_results(folder: str | Path = "output") -> None:
    experiment_results = load_all_experiment_results(folder)
    plot_multiple_experiment_results(experiment_results)


def plot_multiple_experiment_results(experiment_results: list[ExperimentResult]) -> None:
    experiment_results.sort(key=lambda experiment_result: experiment_result.confidence_threshold)
    confidence_thresholds = [
        100 * experiment_result.confidence_threshold for experiment_result in experiment_results
    ]
    accuracy_improvements: list[float] = []
    for experiment_result in experiment_results:
        accuracy_improvements.append(
            100
            * (
                experiment_result.metrics[-1].accuracy_test
                - experiment_result.metrics[0].accuracy_test
            )
        )

    plt.scatter(confidence_thresholds, accuracy_improvements)
    plt.xlabel("Confidence Threshold (%)")
    plt.ylabel("Accuracy Improvements (%pt.)")
    plt.title("Accuracy Improvements vs. Confidence Threshold")
    plt.show()


def plot_accuracy_improvements_per_iteration(experiment_results: list[ExperimentResult]) -> None:
    experiment_results.sort(key=lambda experiment_result: experiment_result.confidence_threshold)
    _, ax = plt.subplots()
    for experiment_result in experiment_results:
        confidence_threshold = experiment_result.confidence_threshold
        metrics = experiment_result.metrics
        # accuracy improvements between consecutive iterations
        accuracy_improvements = [
            (metrics[i + 1].accuracy_test - metrics[i].accuracy_test) * 100
            for i in range(len(metrics) - 1)
        ]
        iterations = list(range(2, len(metrics) + 1))
        plt.plot(
            iterations,
            accuracy_improvements,
            c=plt.cm.viridis(confidence_threshold),
            alpha=0.7,
        )
    # continuous colormap for the legend
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    plt.grid()
    plt.colorbar(sm, ax=ax, label="Confidence Threshold")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy Improvement From Previous Iteration (%pt.)")
    plt.title("Accuracy Improvements per Iteration")
    plt.show()


def plot_high_confidence_accuracy(experiment_results: list[ExperimentResult]) -> None:
    experiment_results.sort(key=lambda experiment_result: experiment_result.confidence_threshold)
    _, ax = plt.subplots()
    n_mnist_train_samples = 60000
    for experiment_result in experiment_results:
        confidence_threshold = experiment_result.confidence_threshold
        metrics = experiment_result.metrics
        data: list[tuple[int, float, float]] = [
            (
                i + 1,
                metrics[i].high_confidence_accuracy * 100,
                # Take a root so the alpha value doesn't decrease too rapidly
                (metrics[i].high_confidence_count / n_mnist_train_samples) ** 0.2,
            )
            for i in range(len(metrics))
            if metrics[i].high_confidence_accuracy is not None
        ]
        if not data:
            continue
        for i in range(len(data) - 1):
            x = [data[i][0], data[i + 1][0]]
            y = [data[i][1], data[i + 1][1]]
            alpha = data[i][2]
            plt.plot(x, y, marker="o", alpha=alpha, c=plt.cm.viridis(confidence_threshold))
    # continuous colormap for the legend
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    plt.grid()
    plt.colorbar(sm, ax=ax, label="Confidence Threshold")
    plt.xlabel("Iteration")
    plt.ylabel("High Confidence Accuracy (%)")
    plt.title("High Confidence Accuracy per Iteration")
    plt.show()


def plot_low_confidence_accuracy(experiment_results: list[ExperimentResult]) -> None:
    experiment_results.sort(key=lambda experiment_result: experiment_result.confidence_threshold)
    _, ax = plt.subplots()
    n_mnist_train_samples = 60000
    for experiment_result in experiment_results:
        confidence_threshold = experiment_result.confidence_threshold
        metrics = experiment_result.metrics
        data: list[tuple[int, float, float]] = [
            (
                i + 1,
                metrics[i].low_confidence_accuracy * 100,
                # Take a root so the alpha value doesn't decrease too rapidly
                (metrics[i].low_confidence_count / n_mnist_train_samples) ** 0.2,
            )
            for i in range(len(metrics))
            if metrics[i].low_confidence_accuracy is not None
        ]
        if not data:
            continue
        for i in range(len(data) - 1):
            x = [data[i][0], data[i + 1][0]]
            y = [data[i][1], data[i + 1][1]]
            alpha = data[i][2]
            plt.plot(x, y, marker="o", alpha=alpha, c=plt.cm.viridis(confidence_threshold))
    # continuous colormap for the legend
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    plt.grid()
    plt.colorbar(sm, ax=ax, label="Confidence Threshold")
    plt.xlabel("Iteration")
    plt.ylabel("Low Confidence Accuracy (%)")
    plt.title("Low Confidence Accuracy per Iteration")
    plt.show()
