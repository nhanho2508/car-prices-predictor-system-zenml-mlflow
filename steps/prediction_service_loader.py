from zenml import step
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str, step_name: str
) -> MLFlowDeploymentService:
    """
    Loads the active MLflow prediction service deployed by a specific pipeline step.

    Parameters:
    - pipeline_name (str): Name of the deployment pipeline
    - step_name (str): Name of the deployment step

    Returns:
    - MLFlowDeploymentService: The active MLflow prediction service instance
    """
    # Get the active MLflow model deployer from the stack
    deployer = MLFlowModelDeployer.get_active_model_deployer()

    # Search for services matching pipeline and step
    services = deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
    )

    if not services:
        raise RuntimeError(
            f"No active MLflow prediction service found for step '{step_name}' "
            f"in pipeline '{pipeline_name}'."
        )

    return services[0]
