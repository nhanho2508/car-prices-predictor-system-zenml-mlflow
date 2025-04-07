import os

from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from pipelines.training_pipeline import ml_pipeline
from steps.dynamic_importer import dynamic_importer
from steps.model_loader import model_loader
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import predictor

# Define path to requirements.txt (used by ZenML if needed for runtime packaging)
requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")


@pipeline
def continuous_deployment_pipeline():
    """
    Trains and deploys an MLflow model using the active ZenML stack.

    - Triggers model training
    - Deploys (or redeploys) the trained model using MLflow
    """
    trained_model = ml_pipeline()  # Optional: capture for other usage
    mlflow_model_deployer_step(
        model=trained_model,
        deploy_decision=True,
        workers=3,
    )


@pipeline(enable_cache=False)
def inference_pipeline():
    """
    Performs batch inference using the deployed MLflow prediction service.

    - Loads data dynamically (e.g., from API or test stub)
    - Loads the active MLflow deployment
    - Sends data to the prediction service
    """
    batch_data = dynamic_importer()

    prediction_service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        step_name="mlflow_model_deployer_step",
    )

    predictor(service=prediction_service, input_data=batch_data)
