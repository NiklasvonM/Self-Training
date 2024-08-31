import matplotlib.pyplot as plt

from .experiment_result import ExperimentResult


def plot_experiment_result(experiment_result: ExperimentResult) -> None:
    metrics = experiment_result.metrics
    iterations = [metric.iteration for metric in metrics]
    test_accuracy = [metric.test_accuracy for metric in metrics]

    plt.plot(iterations, test_accuracy)
    plt.xlabel("Iteration")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs. Iteration")
    plt.show()
