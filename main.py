import os
import json
import yaml
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf

_steps = ["data_cleaning", "data_quality_checks", "data_splits", "train_model", "evaluate_model", "tests"]


# This automatically reads in the configuration
@hydra.main(version_base=None, config_path=".", config_name="config")
def run_pipeline(config: DictConfig):

    # Steps to execute
    steps_par = config.main.pipeline_steps
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    if "data_splits" in active_steps:
        _ = mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "data_splits"),
            entry_point="main",
            parameters={
                "test_size": config.splits_data.test_size,
                "filepath_to_csv_dataset": config.storage.filepath_to_csv_dataset,
                "dirpath_to_data_registry": config.storage.dirpath_to_data_registry,
            },
        )

    if "data_cleaning" in active_steps:
        _ = mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "data_cleaning"),
            entry_point="main",
            parameters={
                "input_csv_data_filepath": config.input_data.input_csv_data_filepath,
                "output_clean_csv_filename": config.output_data.output_clean_csv_filename,
            },
        )

    if "data_quality_checks" in active_steps:
        pass

    if "train_model" in active_steps:
        pass

    if "evaluate_model" in active_steps:
        pass


if __name__ == "__main__":
    run_pipeline()
