import sys
import dvc
import pandas as pd

# from contextlib import contextmanager
from salary_classifier.src.salary_classifier.common.log import logging_config

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
logger = logging_config("logger_data_version")

def load_data_from_dvc(
    path: str,
    repo: str = ".",
    rev: str = None,
    remote: str = None,
    remote_config: dict = None,
    config: dict = None,
    mode: str = "r",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Load a dataset tracked by DVC (Data Version Control) from a specified version.

    This utility provides a reproducible and version-aware data loading mechanism
    for MLOps pipelines. It retrieves a dataset file tracked in a DVC project,
    optionally from a specific Git commit, branch, or tag, and loads it into memory
    as a Pandas DataFrame.

    The function transparently handles local and remote DVC storage and can use
    custom remote configurations (e.g., credentials for S3, GCS, or Azure Blob).

    Args:
        path (str):
            Path to the dataset file, relative to the root of the DVC repository.
        repo (str, optional):
            Path or URL to the DVC repository.
            Use `"."` when calling from within the same project.
            Defaults to `"."`.
        rev (str, optional):
            Git revision identifier (e.g., commit hash, tag, or branch name).
            When omitted, the latest workspace version is used.
        remote (str, optional):
            Name of the DVC remote from which to fetch the data.
            If omitted, DVC uses the default remote or local cache.
        remote_config (dict, optional):
            Dictionary of configuration options to override or extend
            the DVC remote definition (e.g., `{"access_key_id": "...", "secret_key": "..."}`).
        config (dict, optional):
            Dictionary of project-level configuration overrides for DVC.
        mode (str, optional):
            File opening mode (same as built-in `open()`), defaults to `"r"`.
        encoding (str, optional):
            File encoding to use for reading the dataset, defaults to `"utf-8"`.

    Returns:
        pd.DataFrame:
            A Pandas DataFrame containing the dataset loaded from DVC.

    Raises:
        SystemExit:
            If the dataset cannot be loaded (e.g., file not found, invalid credentials,
            corrupted cache, or incompatible revision).

    Notes:
        - This function ensures data reproducibility by linking each ML pipeline step
          to a specific DVC-tracked dataset version.
        - It is recommended to log the `rev` used into MLflow for full lineage tracking.
    """
    try:
        logger.info(
            f"Loading dataset from DVC: path='{path}', repo='{repo}', rev='{rev or 'workspace'}'"
        )

        with dvc.api.open(
            path=path,
            repo=repo,
            rev=rev,
            mode=mode,
            remote=remote,
            remote_config=remote_config,
            config=config,
            encoding=encoding,
        ) as fd:
            df = pd.read_csv(fd)

        logger.info(
            f"Dataset loaded successfully from DVC: {len(df)} rows × {len(df.columns)} columns"
        )
        return df

    except Exception as e:
        logger.exception(
            f"Failed to load dataset from DVC.\n"
            f"Path: {path}\nRepo: {repo}\nRev: {rev}\nError: {e}"
        )
        sys.exit(1)
