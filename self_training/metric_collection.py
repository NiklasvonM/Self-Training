from dataclasses import dataclass


@dataclass
class MetricCollection:
    iteration: int
    number_training_samples: int
    high_confidence_accuracy: float | None
    high_confidence_count: int
    low_confidence_accuracy: float | None
    low_confidence_count: int
    test_accuracy: float

    def __str__(self) -> str:
        result: str = ""
        result += f"Iteration: {self.iteration}\n"
        result += f"Number training samples: {self.number_training_samples}\n"
        result += (
            f"High confidence accuracy: {round(100 * self.high_confidence_accuracy, 2)} %\n"
            if self.high_confidence_accuracy is not None
            else ""
        )
        result += f"High confidence count: {self.high_confidence_count}\n"
        result += (
            f"Low confidence accuracy: {round(100 * self.low_confidence_accuracy, 2)} %\n"
            if self.low_confidence_accuracy is not None
            else ""
        )
        result += f"Low confidence count: {self.low_confidence_count}\n"
        result += f"Test accuracy: {round(100 * self.test_accuracy, 2)} %\n"
        return result
