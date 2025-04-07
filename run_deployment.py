import click
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)


@click.command()
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the running MLflow prediction service",
)
def run_main(stop_service: bool):
    """
    CLI entry point for running or stopping the prices predictor pipeline.
    """
    model_name = "prices_predictor"
    pipeline_name = "continuous_deployment_pipeline"
    step_name = "mlflow_model_deployer_step"

    # Get MLflow model deployer
    deployer = MLFlowModelDeployer.get_active_model_deployer()

    if stop_service:
        services = deployer.find_model_server(
            pipeline_name=pipeline_name,
            pipeline_step_name=step_name,
            model_name=model_name,
            running=True,
        )

        if services:
            services[0].stop(timeout=10)
            print("[bold green] Prediction service stopped successfully.")
        else:
            print("[bold yellow] No running service found to stop.")
        return

    # Run training + deployment
    continuous_deployment_pipeline()

    # Run inference
    inference_pipeline()

    print("\n[bold blue] To inspect experiment runs, launch MLflow UI:[/bold blue]")
    print(f"[italic]    mlflow ui --backend-store-uri {get_tracking_uri()}[/italic]")

    services = deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
    )

    if services and services[0].is_running:
        print("\n[bold green] MLflow Prediction Service is running at:[/bold green]")
        print(f"    [underline]{services[0].prediction_url}[/underline]")
        print(
            "\n To stop it, re-run this command with the [italic yellow]--stop-service[/italic yellow] option."
        )
    else:
        print("[bold red]No active prediction service was found after deployment.")


if __name__ == "__main__":
    run_main()
