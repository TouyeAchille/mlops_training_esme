from typing import Literal
from typing import Dict, Literal
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score
from salary_classifier.src.salary_classifier.common.log import logging_config


# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
logger = logging_config()


def compute_model_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: Literal["train", "test", "val"] = "train",
) -> Dict[str, float]:
    """
    Compute and structure classification performance metrics for ML model evaluation.

    This function is typically called during model evaluation or validation stages.
    It computes precision, recall, and F1-score for a given dataset split (train/test/val),
    and returns the results in a structured format ready for MLflow logging or dashboard tracking.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth (true) labels, binarized or categorical.

    y_pred : np.ndarray
        Predicted labels, binarized or categorical.

    dataset_name : Literal["train", "test", "val"], optional
        Dataset split name for metric tagging. Default is "train".

    Returns
    -------
    Dict[str, float]
        A dictionary containing the computed metrics, e.g.:
        {
            "train_precision": 0.91,
            "train_recall": 0.88,
            "train_f1score": 0.89
        }
    """

    try:
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        f1 = f1_score(y_true, y_pred, zero_division=1)

        metrics = {
            f"{dataset_name}_precision": round(precision, 4),
            f"{dataset_name}_recall": round(recall, 4),
            f"{dataset_name}_f1score": round(f1, 4),
        }
        logger.info("Successfully log metrics: %s", metrics)

        return metrics

    except Exception as e:
        raise ValueError(f"Error computing metrics for dataset '{dataset_name}': {e}")
