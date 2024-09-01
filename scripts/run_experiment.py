from datetime import datetime

from self_training import Experiment, plot_experiment_result, save_experiment_result


def main() -> None:
    experiment = Experiment(initial_subset_size=1000, confidence_threshold=0.98)
    experiment.run(number_iterations=10)
    experiment_result = experiment.get_result()
    save_experiment_result(
        experiment_result,
        f"output/experiment_result_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json",
    )
    plot_experiment_result(experiment_result)


if __name__ == "__main__":
    main()
