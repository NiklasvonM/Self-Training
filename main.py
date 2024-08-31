from self_training import Experiment, plot_experiment_result, save_experiment_result


def main() -> None:
    experiment = Experiment(initial_subset_size=1000, confidence_threshold=0.99)
    experiment_data = experiment.run(number_iterations=10)
    save_experiment_result(experiment_data)
    plot_experiment_result(experiment_data)


if __name__ == "__main__":
    main()
