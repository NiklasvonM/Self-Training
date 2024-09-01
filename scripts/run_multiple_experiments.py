import random
from datetime import datetime

from self_training import Experiment, save_experiment_result


def main() -> None:
    while True:
        threshold = random.uniform(0.0, 1.0)
        print(f"Current threshold: {threshold}")
        experiment = Experiment(initial_subset_size=1000, confidence_threshold=threshold)
        experiment.run(number_iterations=10)
        experiment_result = experiment.get_result()
        save_experiment_result(
            experiment_result,
            f"output/experiment_result_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json",
        )


if __name__ == "__main__":
    main()
