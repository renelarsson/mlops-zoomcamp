import batch # 2.1 See __init__.py
import pandas as pd
from datetime import datetime

# 3.3 Where dt is a helper function
def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

# 3.2 Now create a test and use this as input:
def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)), # Duration OK: 9 minutes
        (1, 1, dt(1, 2), dt(1, 10)), # Duration OK: 8 minutes
        (1, None, dt(1, 2, 0), dt(1, 2, 59)), # Duration < 60 s (0 hours, 0 minutes, 59 seconds)
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)), # Duration > 60 min (1 hour, 0 minutes, 1 second)        
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns) # Return as dataframe
    
    categorical = ['PUlocationID', 'DolocationID']
    # “3.4: Define the expected output and use the assert to make sure that the actual dataframe matches the expected one”
    actual_df = batch.prepare_data(df, categorical) # Pass df to test prepare_data
    expected_rows = [
        ('-1', '-1', 9.0), # Expect 9 min (fillna(-1) for missing location ID values)
        ( '1',  '1', 8.0), # Expect 8 min
    ]

    cols_test = ['PUlocationID', 'DOlocationID', 'duration']
    expected_df = pd.DataFrame(expected_rows, columns=cols_test)
    
    # Assert columns seperately given precision issues
    assert (actual_df['PUlocationID'] == expected_df['PulocationID']).all() # Assert for all values
    assert (actual_df['DOlocationID'] == expected_df['DolocationID']).all() # Assert location
    # Assert absolute difference duration
    assert (actual_df['duration'] - expected_df['duration']).abs().sum() < 0.00001
