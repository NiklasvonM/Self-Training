import dataclasses


@dataclasses.dataclass
class MetricCollection:
    iteration: int
    number_training_samples: int
    high_confidence_count: int
    low_confidence_count: int
    accuracy_test: float
    high_confidence_accuracy: float | None = None
    low_confidence_accuracy: float | None = None
    share_correct_train_labels: float | None = None
    accuracy_in_sample: float | None = None
    accuracy_unlabeled: float | None = None

    def __str__(self) -> str:
        result: str = ""
        result += f"Iteration: {self.iteration}\n"
        result += f"Number training samples: {self.number_training_samples}\n"
        result += (
            f"Share correct train labels: {round(100 * self.share_correct_train_labels, 2)} %\n"
            if self.share_correct_train_labels is not None
            else ""
        )
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
        result += (
            f"Accuracy in sample: {round(100 * self.accuracy_in_sample, 2)} %\n"
            if self.accuracy_in_sample is not None
            else ""
        )
        result += f"Accuracy unlabeled: {round(100 * self.accuracy_unlabeled, 2)} %\n"
        result += f"Test accuracy: {round(100 * self.accuracy_test, 2)} %"
        return result
