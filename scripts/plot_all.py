from self_training.experiment_result import load_all_experiment_results
from self_training.plot import (
    plot_accuracy_improvements_per_iteration,
    plot_multiple_experiment_results,
)

if __name__ == "__main__":
    experiment_results = load_all_experiment_results(scan_subfolders=True)
    plot_multiple_experiment_results(experiment_results)
    plot_accuracy_improvements_per_iteration(experiment_results)
