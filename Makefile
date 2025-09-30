# Makefile for running MLflow projects with different steps
data_clean:
	mlflow run . -P steps=data_cleaning
