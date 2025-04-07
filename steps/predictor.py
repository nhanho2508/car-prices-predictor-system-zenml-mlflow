import json
import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    input_data: str,
) -> np.ndarray:
    """
    Sends input data to a deployed MLflow service for prediction.

    Parameters:
    - service (MLFlowDeploymentService): The active model service.
    - input_data (str): JSON-encoded test data (as string).

    Returns:
    - np.ndarray: Predictions from the model.
    """
    # Ensure the service is running
    service.start(timeout=10)

    # Parse and clean the JSON input
    payload = json.loads(input_data)
    #payload.pop("columns", None)
    #payload.pop("index", None)

    expected_cols = [
        "name",
    "year",
    "selling_price",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "seats"
    ]

    # Convert JSON to DataFrame and prepare input
    # df = pd.DataFrame(payload["data"], columns=expected_cols)
    # model_input = df.to_dict(orient="records")

    # # Run prediction
    # prediction = service.predict(model_input)

    # Convert the data into a DataFrame with the expected columns
    df = pd.DataFrame(payload["data"], columns=expected_cols)

    # Convert DataFrame to JSON list for prediction
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data_array = np.array(json_list)

    # Run the prediction
    prediction = service.predict(data_array)

    return np.array(prediction)
