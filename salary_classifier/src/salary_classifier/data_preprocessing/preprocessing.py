"""
Module: data_preprocessing.py
Description:
    This module handles preprocessing of raw tabular data as part of an ML pipeline.
    It cleans the dataset, applies standard transformations, and logs artifacts
    and metadata to MLflow for traceability.
"""

# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
import os
import sys
from pathlib import Path
import argparse
import mlflow
import dvc.api
import pandas as pd

from salary_classifier.src.salary_classifier.common.log import logging_config
from salary_classifier.src.salary_classifier.common.utils import load_data_from_dvc

# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------
logger = logging_config("log_preprocessing")


# -------------------------------------------------------------------
# Data preprocessing function
# -------------------------------------------------------------------
def process_data(args):
    """Clean a raw CSV dataset, save it, and log it to MLflow.

    This function performs a sequence of preprocessing operations and
    tracks the cleaned dataset as an artifact within an MLflow run.

    Steps performed:
        1. Load the raw CSV dataset from the provided input path.
        2. Clean column names (strip whitespace).
        3. Drop duplicates and rows with missing values.
        4. Save the cleaned dataset locally.
        5. Log the cleaned dataset and metadata to MLflow.

    Args:
        args (argparse.Namespace):
            input_data (str): Path to the raw CSV file.
            output_data (str): Filename for the cleaned output CSV.

    Raises:
        SystemExit: If the dataset cannot be loaded or saved properly.

    Returns:
        pd.DataFrame: The cleaned dataframe.

    """

    logger.info(" Starting data preprocessing pipeline...")

    logger.info("Load dataset...")
    df=load_data_from_dvc(path=args.input_data, rev="v1.0.0")

    logger.info("Cleaning dataset...")
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    df = df.dropna()
    logger.info(" Data cleaning complete.")

    
    logger.info('Save cleaned data...')
    output_dir = Path(__file__).resolve().parents[4] / "datastores" / "cleaned_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / args.output_data

    logger.info(f"Saving cleaned data to: {output_path}")


    logger.info("Step 4: MLflow logging")
    with mlflow.start_run() as parent_run:
        logger.info(f"parent_run_id: {parent_run.info.run_id}")
        with mlflow.start_run(run_name="preprocess_step", nested=True) as child_run:
            logger.info("child_run_id : %s", child_run.info.run_id)
            try:
                df.to_csv(output_path, index=False)
                mlflow.set_tag("step", "preprocessing")
                mlflow.log_param("input_data", args.input_data)
                mlflow.log_param("output_data", str(output_path))
                logger.info("Cleaned dataset successfully logged to MLflow.")
            except Exception as e:
                logger.exception("Failed to log cleaned data to MLflow.")
                sys.exit(1)

    logger.info("Data preprocessing completed successfully.")

    return df


def parsers_arguments():
    """
    Parse command-line arguments for the data cleaning stage.

    Returns:
        argparse.Namespace: Contains input and output file paths.
    """
    parser = argparse.ArgumentParser(description="data cleaning stage")

    parser.add_argument(
        "--input_data",
        type=str,
        help="Absolute or relative path to the raw input CSV file.",
        required=True,
    )

    parser.add_argument(
        "--output_data",
        type=str,
        help="file path to save clean data ",
        required=True,
        default="clean_data.csv",
    )

    args = parser.parse_args()

    return args


def main():
    args = parsers_arguments()
    # run function
    process_data(args)


if __name__ == "__main__":
    main()
