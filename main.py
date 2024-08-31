from self_training import Experiment


def main() -> None:
    experiment = Experiment()
    experiment.run(number_iterations=10)


if __name__ == "__main__":
    main()
