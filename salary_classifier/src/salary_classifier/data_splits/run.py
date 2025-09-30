import argparse
import logging
import mlflow
import pandas as pd

from pathlib import Path
from typing import Optional
from argparse import Namespace
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# create and active experiment
mlflow.set_experiment("ml_pipeline_clf")


def _save_and_log_artifact(
    df: pd.DataFrame, filename: str, artifact_path: Optional[str] = None
) -> Path:
    """Save DataFrame to CSV and log as MLflow artifact."""
    file_path = Path(filename)
    df.to_csv(file_path, index=False)
    mlflow.log_artifact(str(file_path), artifact_path=artifact_path)
    return file_path


def split_data(args: Namespace) -> None:
    """
    Splits data into train, test, and validation sets.

    Args:
        args: Argument namespace containing filepath_to_csv_dataset and test_size.

    Returns:
        None
    """
    logger.info("Reading CSV dataset...")
    try:
        df = pd.read_csv(args.filepath_to_csv_dataset)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")

    logger.info("Stripping whitespace from column names")
    df.columns = df.columns.str.strip()

    logger.info("Splitting data into train and test sets")
    data_train, test_data = train_test_split(
        df,
        test_size=args.test_size,
        random_state=42,
        shuffle=True,
        stratify=df["salary"],
    )

    logger.info("Splitting train data into train and validation sets")
    train_data, val_data = train_test_split(
        data_train,
        test_size=args.test_size,
        random_state=42,
        shuffle=True,
        stratify=data_train["salary"],
    )

    # data Registry Path
    logger.info("Save output artifact file into local dvc data registry")
    dir_path = Path(args.dirpath_to_data_registry)
    dir_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    # MLflow Logging
    with mlflow.start_run(run_name="split_data"):
        try:
            logger.info("Saving and logging artifacts...")
            file_paths = {
                "train_data.csv": train_data,
                "test_data.csv": test_data,
                "val_data.csv": val_data,
            }
            for filename, data in file_paths.items():
                file_path = _save_and_log_artifact(
                    data, dir_path / filename, "splits_data"
                )
                file_path.unlink()  # Delete file after logginig
                logger.info(f"Deleted temporary file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to log artifacts: {e}")


def parsers_arguments():
    parser = argparse.ArgumentParser(description="save split data")

    parser.add_argument(
        "--filepath_to_csv_dataset",
        type=str,
        help="file path to raw csv dataset",
        required=True,
    )

    parser.add_argument(
        "--dirpath_to_data_registry",
        type=str,
        help="path to save output artifact file in your  data registry (storage)",
        required=True,
    )

    parser.add_argument(
        "--test_size",
        type=float,
        help="float, should be between 0.0 and 1.0 and represent the proportion of the dataset",
        default=0.1,
        required=False,
    )

    args = parser.parse_args()

    return args


def main():
    args = parsers_arguments()

    # run function
    split_data(args)


if __name__ == "__main__":
    main()
