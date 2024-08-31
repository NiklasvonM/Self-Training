from datetime import datetime

import numpy as np

from self_training import Experiment, save_experiment_result


def main() -> None:
    confidence_thresholds: list[float] = reversed(np.linspace(0.2, 1.0, 10).tolist())
    for threshold in confidence_thresholds:
        experiment = Experiment(initial_subset_size=1000, confidence_threshold=threshold)
        experiment.run(number_iterations=10)
        experiment_result = experiment.get_result()
        save_experiment_result(
            experiment_result,
            f"output/experiment_result_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json",
        )


if __name__ == "__main__":
    main()
