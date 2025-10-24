# Variables ---------------------------------------------------
INPUT_DATA_PATH ?= ../../../../datastores/raw_data/census.csv
OUTPUT_DATA_FILENAME ?= cleaned_data.csv
EXPERIMENT_NAME ?= preprocessing
preprocess:
	@echo "Running data preprocessing step with python..."
	python preprocessing.py --input_data $(INPUT_DATA_PATH) --output_data $(OUTPUT_DATA_FILENAME)
	@echo "Data preprocessing completed."

preprocessing:
	@echo "Running data preprocessing step with mlflow run..."
	mlflow run . -P input_data=$(INPUT_DATA_PATH) -P output_data=$(OUTPUT_DATA_FILENAME) -b local --experiment-name $(EXPERIMENT_NAME)
	@echo "Data preprocessing completed."	