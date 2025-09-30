import argparse
import json
import pandas as pd
import logging
import mlflow

from sklearn.impute import SimpleImputer
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn.metrics import fbeta_score, precision_score, recall_score
from common.metrics import compute_model_metrics


# create and active mlflow experiment
mlflow.set_experiment("ml_pipeline_clf")

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def train_ml_model(args):
    """
    Train ML model.

    Args:
        args: Argument namespace containing filepath_to_train_csv_dataset and filepath_to_test_csv_dataset.

    Returns:
        trained model
    """

    # Read train and test CSV datasets
    logger.info("Reading train CSV dataset")
    df = pd.read_csv(args.filepath_to_train_csv_dataset)

    logger.info("Reading test CSV dataset")
    df = pd.read_csv(args.filepath_to_test_csv_dataset)

    # Split data into features and target
    logger.info("Splitting train data into features and target")
    X_train = df.drop(
        labels="salary", axis=1
    )  # features columns (numerical and categorical)
    y_train = df["salary"]  # target column

    logger.info("Splitting test data into features and target")
    X_test = df.drop(
        labels="salary", axis=1
    )  # features columns (numerical and categorical)
    y_test = df["salary"]  # target column

    logger.info("process features...")

    # Select columns from DataFrame
    logger.info("Selecting columns from DataFrame")
    categorical_columns = make_column_selector(dtype_include=object)(X_train)
    numerical_columns = make_column_selector(dtype_exclude=object)(X_train)

    # Define preprocessor for numerical columns
    logger.info("Defining preprocessor for numerical columns (features)")
    numerical_preprocessor = make_pipeline(
        SimpleImputer(strategy="most_frequent"), StandardScaler()
    )

    # Define preprocessor for categorical columns
    logger.info("Defining preprocessor for categorical columns (features)")
    categorical_preprocessor = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore",sparse_output=False),
    )

    # combine feature preprocessor for numerical and categorical columns into one
    logger.info("Combining preprocessor for numerical and categorical columns")
    full_preprocessor = ColumnTransformer(
        transformers=[
            ("numerical_preprocessor", numerical_preprocessor, numerical_columns),
            ("categorical_preprocessor", categorical_preprocessor, categorical_columns),
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    logger.info("process target: label binarizer")
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test  = lb.transform(y_test)

    # Define the model hyperparameters
    with open(args.model_params) as fp:
        model_params = json.load(fp)

    # Define binary classifier ml model
    logger.info("Defining ML model")
    ml_model = args.model_name(**model_params)

    # Define ml pipeline
    logger.info("Defining ML pipeline")
    ml_pipeline = Pipeline(
        steps=[("preprocessor", full_preprocessor), ("clf_model", ml_model)]
    )

    # Train model
    logger.info("Training model")
    ml_pipeline.fit(X_train, y_train)

    # Evaluate model on train data
    logger.info("Evaluating model on train data")
    y_train_pred = ml_pipeline.predict(X_train) 

    # Evaluate model on test data
    logger.info("Evaluating model on test data")
    y_test_pred= ml_pipeline.predict(X_test)

    # Calculate model metrics
    logger.info("Calculating model metrics")
    train_score: dict = compute_model_metrics(y_train, y_train_pred)
    test_score : dict = compute_model_metrics(y_test, y_test_pred)

    # start mlflow run
    with mlflow.start_run(run_name="train_ml_clf") as run:

        mlflow.set_tag("Training Info :", "Binary Classifier model for census data")

        # Log model parameters
        logger.info("Logging model parameters")
        mlflow.log_params(model_params)

        # Log model metrics
        logger.info("Logging model metrics")
        mlflow.log_metrics("train_score", train_score)
        mlflow.log_metrics("test_score", test_score)

        # Infer the model signature
        signature = infer_signature(X_train, ml_pipeline.predict(X_train))

        # Log model
        logger.info("Logging model")
        mlflow.sklearn.log_model(sk_model=ml_pipeline, 
                                 artifact_path="model",
                                 signature=signature,
                                 input_example=X_train.head(5),
                                 registered_model_name="model"
                                 )

    return ml_pipeline, lb, 
