import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from .errors import NoSavedExperimentError
from .metric_collection import MetricCollection


@dataclasses.dataclass
class ExperimentResult:
    confidence_threshold: float
    metrics: list[MetricCollection]


def save_experiment_result(
    experiment_result: ExperimentResult, filename: str | Path = "output/experiment_result.json"
) -> None:
    output_path = Path(filename)
    os.makedirs(output_path.parent, exist_ok=True)
    serialized_result = dataclasses.asdict(experiment_result)
    with open(filename, "w", encoding="UTF-8") as f:
        json.dump(serialized_result, f, ensure_ascii=False, indent=4)
    print(f"Saved experiment result to {filename}.")


def load_experiment_result(
    filename: str | Path = "output/experiment_result.json",
) -> ExperimentResult:
    try:
        with open(filename, encoding="UTF-8") as f:
            data: dict[str, Any] = json.load(f)
        metrics = [MetricCollection(**metric) for metric in data["metrics"]]
        result = ExperimentResult(
            confidence_threshold=data["confidence_threshold"], metrics=metrics
        )
        return result
    except Exception as e:
        raise NoSavedExperimentError(
            "Failed to load metrics. Have you run the experiment and saved the metrics?"
        ) from e


def load_all_experiment_results(
    folder: str | Path = "output", scan_subfolders: bool = False
) -> None:
    experiment_results: list[ExperimentResult] = []
    folder = Path(folder)
    paths = folder.rglob("*.json") if scan_subfolders else folder.glob("*.json")
    for filename in paths:
        try:
            experiment_result = load_experiment_result(filename)
            experiment_results.append(experiment_result)
        except NoSavedExperimentError:
            pass
    return experiment_results
