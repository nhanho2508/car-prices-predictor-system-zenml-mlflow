from sklearn.pipeline import Pipeline
from zenml import Model, step


@step
def model_loader(model_name: str) -> Pipeline:
    """
    Load a production-ready scikit-learn model pipeline by name.

    Parameters:
    - model_name (str): The registered name of the model in ZenML.

    Returns:
    - Pipeline: The loaded pipeline artifact (e.g., sklearn pipeline).
    """
    # Reference the model in production
    model = Model(name=model_name, version="production")

    # Load the pipeline artifact (default artifact name: "sklearn_pipeline")
    pipeline: Pipeline = model.load_artifact("sklearn_pipeline")

    return pipeline
