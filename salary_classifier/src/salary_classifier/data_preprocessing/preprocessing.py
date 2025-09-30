import sys
import os
import tempfile
from pathlib import Path
import argparse
import mlflow
import pandas as pd


# ---------------------------------------------------------------------
# Add the parent directory to the system path to import common utilities
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.utils import logging_config

# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------
logger = logging_config()

# ---------------------------------------------------------------------
# Set the MLflow tracking URI to a local directory
# This allows MLflow to log artifacts and metadata locally.
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Set the MLflow experiment for this pipeline stage
# ---------------------------------------------------------------------
# If there is an active MLflow run, end it before starting a new one.

mlflow.set_experiment("ML_Pipeline-On-Census-Data")


# ---------------------------------------------------------------------
# Main data cleaning function
# ---------------------------------------------------------------------
def process_data(args):
    """
    Cleans a CSV dataset by applying standard preprocessing steps and logs the cleaned data as an MLflow artifact.

    Steps performed:
    - Reads the raw CSV file from the given input path.
    - Strips whitespace from column names.
    - Drops duplicate rows.
    - Drops rows containing missing values.
    - Saves the cleaned data to a temporary file.
    - Logs the cleaned data as an artifact to MLflow within a run.

    Args:
        args: An argparse.Namespace object containing:
            - input_data (str): Path to the raw CSV file to clean.
            - output_data (str): Filename for the cleaned CSV artifact.

    Raises:
        SystemExit: If the input file cannot be read or if logging fails.

    Example usage:
        args = argparse.Namespace(
            input_csv_data_filepath="data/raw.csv",
            output_clean_csv_filename="cleaned.csv"
        )
        process_data(args)
    """

    logger.info("Reading CSV dataset...")

    try:
        df = pd.read_csv(args.input_data)
        logger.info(f"CSV file read successfully: {args.input_data}")
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        sys.exit(1)

    # Cleaning steps
    logger.info("Stripping whitespace from column names")
    df.columns = df.columns.str.strip()

    logger.info("Dropping duplicate rows")
    df = df.drop_duplicates()

    logger.info("Dropping rows with missing values")
    df = df.dropna()

    # Save cleaned data to a temporary file and log as artifact
    logger.info("Creating a temporary directory to save cleaned data")
    #os.makedirs("outputs", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_filepath = Path(tmp_dir) / args.output_data
        logger.info(f"Temporary file path for cleaned data: {temp_filepath}")

        with mlflow.start_run(run_name="process_data") as run:
            try:
                df.to_csv(temp_filepath, index=False)
                mlflow.log_artifact(temp_filepath)
                mlflow.set_tag("stage", "data_cleaning")
                mlflow.log_param("input_csv_data_filepath", args.input_data)
                mlflow.log_param("output_clean_csv_filename", args.output_data)
                logger.info(f"Cleaned data logged as MLflow artifact: {temp_filepath}")
            except Exception as e:
                logger.error(f"Failed to log cleaned data as MLflow artifact: {e}")
                sys.exit(1)

    logger.info("Data cleaning process completed successfully.")


# ---------------------------------------------------------------------
# Command-line argument parser for the data cleaning stage
# ---------------------------------------------------------------------
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
        required=False,
        default="cleaned_data.csv",
    )

    args = parser.parse_args()

    return args


def main():
    args = parsers_arguments()

    # run function
    process_data(args)


if __name__ == "__main__":
    main()
