import pandas as pd
from zenml import step


@step(enable_cache=False)
def dynamic_importer() -> str:
    """
    Simulates dynamic data import for testing purposes.

    In a production context, this could be replaced by an API call, database read, or file loader.
    
    Returns:
    - str: A JSON-formatted string of sample data in split orientation.
    """
    sample_data = {
        "name": ["Maruti Swift", "Skoda Rapid", "Honda City", "Hyundai i20"],
        "year": [2014, 2014, 2006, 2010],
        "selling_price": [450000, 370000, 158000, 225000],
        "km_driven": [145500, 120000, 140000, 127000],
        "fuel": ["Diesel", "Diesel", "Petrol", "Diesel"],
        "seller_type": ["Individual"] * 4,
        "transmission": ["Manual"] * 4,
        "owner": ["First Owner", "Second Owner", "Third Owner", "First Owner"],
        "mileage": ["23.4 kmpl", "21.14 kmpl", "17.7 kmpl", "23.0 kmpl"],
        "engine": ["1248 CC", "1498 CC", "1497 CC", "1396 CC"],
        "max_power": ["74 bhp", "103.52 bhp", "78 bhp", "90 bhp"],
        "seats": [5, 5, 5, 5],
    }


    df = pd.DataFrame(sample_data)
    return df.to_json(orient="split")
