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


@step(enable_cache=False)
def feature_engineering_step(
    df: pd.DataFrame, strategies: list
) -> pd.DataFrame:
    """
    Applies a selected feature engineering strategy to specified columns.

    Parameters:
    - df (pd.DataFrame): Input dataset
    - strategies (str): Strategy to apply ('log', 'standard_scaling', 'minmax_scaling', 'onehot_encoding',.....)


    Returns:
    - pd.DataFrame: Transformed dataset
    """

    custom_strategies = OrderedDict([
        ('extract_column', SplitExtractAndDrop(source_column='name', new_column='brand')),
        ('column_difference', ColumnReplacerWithDifference(constant=2025, column='year', new_name='age')),
        ('drop_column', ColumnDropper(columns=['year'])),
        ('map_value', ValueMapper('owner', {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3})),
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
        ('log_transform', LogTransformation(features=['selling_price', 'max_power', 'age'])),

    ])

    

    
    df_transformed = df.copy()

    # Thực hiện theo đúng thứ tự của custom_strategies nếu key có trong strategies
    for key, strategy in custom_strategies.items():
        if key in strategies:
            transformer = FeatureEngineer(strategy)
            df_transformed = transformer.apply_feature_engineering(df_transformed)

    return df_transformed
