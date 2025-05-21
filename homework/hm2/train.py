import os
import pickle
import click
# Import MLflow
import mlflow

from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error

# Set tracking URI to local sqlite db
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Set experiment name if it doesn't exist
mlflow.set_experiment("train-random-forest")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    
    # Enable autologging     
    mlflow.sklearn.autolog()
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    
    # Wrap the training code to associate it with the current run
    with mlflow.start_run():

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        # rmse = mean_squared_error(y_val, y_pred, squared=False)
        rmse = root_mean_squared_error(y_val, y_pred)

if __name__ == '__main__':
    run_train()
