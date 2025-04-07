import click
from pipelines.training_pipeline import ml_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


@click.command()
def main():
    """
    Run the ML training pipeline and print instructions to launch MLflow UI.
    """
    click.secho(" Running training pipeline...", fg="green")
    run = ml_pipeline()

    # Optional: access the trained model artifact (uncomment if needed)
    # model = run["model_building_step"]
    # click.secho(f"Trained model type: {type(model)}", fg="blue")

    mlflow_uri = get_tracking_uri()
    click.secho("\n To inspect your MLflow runs, open the UI:", fg="cyan", bold=True)
    click.secho(f"    mlflow ui --backend-store-uri '{mlflow_uri}'\n", fg="yellow")
    click.secho("Runs are tracked in your configured experiment within MLflow.", fg="cyan")


if __name__ == "__main__":
    main()
