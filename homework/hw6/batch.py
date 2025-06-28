#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
import pandas as pd

# 3.1 “We'll start with the pre-processing logic inside read_data… create a function prepare_data”
def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

# 1.2. “Move all the code (except read_data) inside main” – will not be tested
def read_data(filename, categorical): # 1.3a. “Make categorical a parameter for read_data…” (→ 1.3b) 

# 4.6 “Reading from Localstack S3 with Pandas… specify the endpoint url:”
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

    if S3_ENDPOINT_URL:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }

        df = pd.read_parquet(filename, storage_options=options) # Overwrite endpoint
    else:
        df = pd.read_parquet(filename) # Default endpoint

    return prepare_data(df, categorical)

# 6.1 “Create a function save_data which works similarly to read_data… for saving a dataframe.”
def save_data(filename, df):
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

    if S3_ENDPOINT_URL:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }

        df.to_parquet(filename, engine='pyarrow', index=False, storage_options=options)
    else:
        df.to_parquet(filename, engine='pyarrow', index=False)

def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    # 4.4 “Make input and output paths configurable” (If not specified, use default - else overwrite)
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    # 4.4 “Make input and output paths configurable”
    default_output_pattern = 's3://mlflow-models-rll/yellow_tripdata_/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

# 1.1 “Let's create a function main with two parameters: year and month:”
def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)    
   # “To make it easier to run it, you can write results to your local filesystem. E.g. here:”
   # output_file = f'taxi_type=yellow_year={year:04d}_month={month:02d}.parquet'	

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']
    
    df = read_data(input_file, categorical) # 1.3b. “…and pass it inside main” (1.3a →) 
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('Predicted mean duration:',y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    save_data(output_file, df_result)

# 1.4 “Now we need to create the "main" block from which we'll invoke the main function.”
if __name__ == '__main__': 
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)
