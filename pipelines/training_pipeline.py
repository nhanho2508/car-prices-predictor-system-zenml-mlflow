from zenml import pipeline, Model

from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_splitter_step
from steps.feature_engineering_step import feature_engineering_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from steps.outlier_detection_step import outlier_detection_step


@pipeline(
    model=Model(name="prices_predictor")
)
def ml_pipeline():
    """
    Full end-to-end ML pipeline:
    - Ingest data from ZIP
    - Handle missing values
    - Apply feature engineering
    - Remove outliers
    - Split data
    - Train model
    - Evaluate performance
    """
    # 1. Ingest raw data
    raw_data = data_ingestion_step(
        file_path="/mnt/c/Users/ADMIN/source/course/mlops_course/prices-predictor-system-mlflow-zenml/data/archive.zip"
    )
 
    # 2. Handle missing values
    filled_data = handle_missing_values_step(raw_data, "drop")

    # 3. Feature engineering
    engineered_data = feature_engineering_step(filled_data,
        strategies = ['extract_column',
    'column_difference',
    'drop_column',
    'map_value',
    'strip_units',
    'type_cast',
    'log_transform',
    'one_hot_encode']
    )

    # 4. Outlier removal
    clean_data = outlier_detection_step(
        df=engineered_data,
        column_name="selling_price"
    )

    clean_data = outlier_detection_step(
        df=clean_data,
        column_name="km_driven"
    )

    clean_data = outlier_detection_step(
        df=clean_data,
        column_name="mileage"
    )

    clean_data = outlier_detection_step(
        df=clean_data,
        column_name="max_power"
    )

    # 5. Split dataset
    X_train, X_test, y_train, y_test = data_splitter_step(
        df=clean_data,
        target_column="selling_price"
    )

    # 6. Train model
    model = model_building_step(
        X_train=X_train,
        y_train=y_train
    )

    # 7. Evaluate model
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model,
        X_test=X_test,
        y_test=y_test
    )

    return model
