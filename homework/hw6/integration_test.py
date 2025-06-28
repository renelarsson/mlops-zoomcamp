import os
import batch
import pandas as pd
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

options = {
    'client_kwargs': {
        'endpoint_url': S3_ENDPOINT_URL
    }
}

# 5.2 “We'll use the dataframe we created in Q3 (the dataframe for the unit test) and save it to S3.”
data = [
    (None, None, dt(1, 1), dt(1, 10)), 
    (1, 1, dt(1, 2), dt(1, 10)), 
    (1, None, dt(1, 2, 0), dt(1, 2, 59)), 
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),   
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

input_file = batch.get_input_path(2023, 1)
output_file = batch.get_output_path(2023, 1)

# 5.3 “Use this snipped for saving the file:”
df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

# 6.2  “Let's run the batch.py… we can use os.system for doing that… saves the result to localstack”
os.system('python3 batch.py 2023 1')

# 6.3 “What's the sum of predicted durations for the test dataframe?”
actual_df = pd.read_parquet(output_file, storage_options=options)
print(actual_df)
