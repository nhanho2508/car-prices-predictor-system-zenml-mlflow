import pandas as pd
from collections import OrderedDict
from .src.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    MinMaxScaling,
    OneHotEncoding,
    StandardScaling,
    ColumnDifference,
    ColumnDropper,
    ColumnReplacerWithDifference,
    TypeCaster,
    UnitRemover,
    ValueMapper,
    SplitExtractAndDrop

)
from zenml import step


@step
def feature_engineering_step(
    df: pd.DataFrame, strategies: list
) -> pd.DataFrame:
    """
    Applies a selected feature engineering strategy to specified columns.

    Parameters:
    - df (pd.DataFrame): Input dataset
    - strategy (str): Strategy to apply ('log', 'standard_scaling', 'minmax_scaling', 'onehot_encoding',.....)
    - features (list): List of feature column names to transform

    Returns:
    - pd.DataFrame: Transformed dataset
    """
    if not features:
        features = []

    custom_strategies = OrderedDict([
        ('extract_brand', SplitExtractAndDrop(source_column='name', new_column='brand')),
        ('age', ColumnReplacerWithDifference(constant=2025, column='year')),
        ('drop_year', ColumnDropper(columns=['year'])),
        ('map_owner', ValueMapper('owner', {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3})),
        ('strip_units', UnitRemover({
            'mileage': r'(kmpl|km/kg)',
            'engine': r'CC',
            'max_power': r'b(h)?p'
        })),
        ('type_cast', TypeCaster({
            'mileage': 'float',
            'engine': 'float',
            'max_power': 'float',
            'seats': 'str'
        })),
        ('log_transform', LogTransformation(features=['selling_price', 'max_power', 'age']))
    ])

    

    
    df_transformed = df.copy()

    # Thực hiện theo đúng thứ tự của custom_strategies nếu key có trong strategies
    for key, strategy in custom_strategies.items():
        if key in strategies:
            transformer = FeatureEngineer(strategy)
            df_transformed = transformer.apply_feature_engineering(df_transformed)

    return df_transformed
