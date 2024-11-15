from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import pickle
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from google.oauth2 import service_account
import lightgbm as lgb
import pandas as pd
import pandas_gbq
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import xgboost as xgb
import yaml # pip install pyyaml


class MLModelTrainer():
    """Class to train, evaluate and save machine learning models using XGBoost and LightGBM."""

    def __init__(self) -> None:
        """Initialize the class by loading environment variables and setting up credentials."""

        # Load environment variables
        load_dotenv()

        self.bigquery_project_id = os.getenv("BIGQUERY_PROJECT_ID")
        self.bigquery_ml_input_table = os.getenv("BIGQUERY_ML_INPUT_TABLE")
        self.credentials = service_account.Credentials.from_service_account_file(
            "service-account-key.json"
        )

        # Check for presence of required environment variables
        if not all([self.bigquery_project_id, self.bigquery_ml_input_table, self.credentials]):
            raise ValueError("Missing required environment variables.")
        
    def load_features_target_config(self, config_path: str = "features_target_config.yaml") -> Dict[str, List[str]]:
        """Load model features and target configuration from a YAML file.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            Dict[str, List[str]]: Configuration dictionary with features and target.
        """
    
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        return config

    def build_sql_query(self, columns: List[str], bigquery_project_id: str, bigquery_table: str) -> str:
        """Build SQL query to fetch data from BigQuery.

        Args:
            columns (List[str]): List of column names to fetch.

        Returns:
            str: SQL query string.
        """
        
        columns_str = ", ".join(columns)
        
        return f"""
        select
            {columns_str}
        from
            `{bigquery_project_id}.{bigquery_table}`
        """
    
    def read_bigquery_table(self, sql_query: str) -> pd.DataFrame:
        """Read data from BigQuery table.
        
        Args:
            sql_query (str): SQL query to execute.
            
        Returns:
            pd.DataFrame: Query results as DataFrame.
        """

        df = pandas_gbq.read_gbq(
            sql_query,
            project_id=self.bigquery_project_id,
            credentials=self.credentials
        )

        return df
    
    def add_publish_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a 'publish_date' column to the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with 'publish_date' column.
        """
        
        df_copy = df.copy()
        
        df_copy["publish_date"] = pd.to_datetime(
            dict(
                year=df_copy["publish_year"], 
                month=df_copy["publish_month"], 
                day=df_copy["publish_day"]
            )
        )

        return df_copy

    def find_cutoff_date(self, df: pd.DataFrame, cutoff_days: int = 180) -> pd.Timestamp:
        """Find the cutoff date based on the given number of days.

        Args:
            df (pd.DataFrame): Input DataFrame.
            cutoff_days (int): Number of days to look back.

        Returns:
            pd.Timestamp: Cutoff date.
        """
        
        # Calculate the cutoff date (n days before the latest date in the dataset)
        latest_date = df["publish_date"].max()
        cutoff_date = latest_date - timedelta(days=cutoff_days)
        
        return cutoff_date
    
    def split_data_by_date(
        self,
        df: pd.DataFrame,
        cutoff_days: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets based on a cutoff date.

        Args:
            df (pd.DataFrame): Input DataFrame.
            cutoff_days (int): Number of days to use for cutoff.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test DataFrames.
        """
        
        df_copy = df.copy()
        cutoff_date = self.find_cutoff_date(df_copy, cutoff_days)
        
        # Create train and test DataFrames based on the cutoff date
        df_train = df_copy.query("publish_date <= @cutoff_date").reset_index(drop=True)
        df_test = df_copy.query("publish_date > @cutoff_date").reset_index(drop=True)
        
        return df_train, df_test

    def split_train_test(
            self,
            df_train: pd.DataFrame,
            df_test: pd.DataFrame,
            model_features: List[str],
            model_target: str
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Extract features and target variables from the train and test DataFrames.

        Args:
            df_train (pd.DataFrame): DataFrame containing the training set.
            df_test (pd.DataFrame): DataFrame containing the test set.
            model_features (List[str]): List of feature column names.
            model_target (str): The target column name.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing:
                - X_train (pd.DataFrame): Features for the training set.
                - X_test (pd.DataFrame): Features for the test set.
                - y_train (pd.Series): Target values for the training set.
                - y_test (pd.Series): Target values for the test set.
        """
        
        X_train = df_train[model_features]
        X_test = df_test[model_features]
        y_train = df_train[model_target]
        y_test = df_test[model_target]

        return X_train, X_test, y_train, y_test

    def evaluate_model(self, y_true: pd.Series, y_pred: List[float]) -> Tuple[float, float]:
        """
        Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

        Args:
            y_true (pd.Series): Actual values.
            y_pred (List[float]): Predicted values.

        Returns:
            Tuple[float, float]: MAE and RMSE scores.
        """

        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        
        return mae, rmse
    
    def calculate_baseline(self, df_train: pd.DataFrame, days: int) -> int:
        """Calculate baseline prediction using the average view count over the past n days.

        Args:
            df_train (pd.DataFrame): Training DataFrame.
            days (int): Number of days to look back.

        Returns:
            float: Average view count as baseline.
        """
        
        # Calculate the cutoff date (n days before the latest date in the dataset)
        latest_date = df_train["publish_date"].max()
        cutoff_date = latest_date - timedelta(days=days)
        
        # Filter data to only include the past n days
        recent_data = df_train[df_train["publish_date"] > cutoff_date]
        
        if not recent_data.empty:
            return round(recent_data["view_count"].mean())
        else:
            raise ValueError("No data available for the past n days.")

    def baseline_model_predictions(self, df_train: pd.DataFrame, y_test: pd.Series, days: int = 30) -> List[float]:
        """
        Predict baseline values using the average view count over a specified number of days.

        Args:
            df_train (pd.DataFrame): The training DataFrame containing historical data.
            y_test (pd.Series): The true target values for the test set.
            days (int, optional): The number of days to look back for calculating the average.
                                Defaults to 30 days.

        Returns:
            List[float]: A list of baseline predictions, with each value being the same (average view count).
        """

        baseline_value = self.calculate_baseline(df_train, days)
        predictions = [baseline_value] * len(y_test)

        return predictions
    
    def build_xgboost_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
        """Train an XGBoost model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.

        Returns:
            xgb.XGBRegressor: Trained XGBoost model.
        """

        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)

        # Feature importance visualization
        fig, ax = plt.subplots(figsize=(8, 4))
        xgb.plot_importance(model, importance_type="gain", ax=ax, title="XGBoost feature importance")
        plt.show()

        return model
    
    def build_lightgbm_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMRegressor:
        """Train a LightGBM model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.

        Returns:
            lgb.LGBMRegressor: Trained LightGBM model.
        """

        model = lgb.LGBMRegressor()
        model.fit(X_train, y_train)

        # Feature importance visualization
        lgb.plot_importance(model, importance_type="gain", figsize=(8, 4), title="LightGBM feature importance")
        plt.show()
        
        return model
    
    def save_model(self, model: Any, filename: str) -> None:
        """
        Save a trained machine learning model to a file with a timestamp in the filename.
        
        Args:
            model (Any): The trained model object to be saved.
            filename (str): The filename (excluding the timestamp and .pkl extension).
        """

        timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_with_timestamp = f"{filename}_{timestamp_now}.pkl"
        
        with open(filename_with_timestamp, "wb") as file:
            pickle.dump(model, file)
    
    def run(self, cutoff_days: int = 180) -> Tuple[float, ...]:
        """Main method to run the end-to-end pipeline."""
        
        # Define basic variables
        config = self.load_features_target_config()
        model_features = config["model_features"]
        model_target = config["model_target"][0]
        columns = [*config["model_features"], *config["model_target"]]

        # Set up the dataframe used for training
        sql_query = self.build_sql_query(columns, self.bigquery_project_id, self.bigquery_ml_input_table)
        df_ml_input = self.read_bigquery_table(sql_query)
        df_ml_input_updated = self.add_publish_date_column(df_ml_input)

        #  Split data into train and test sets based on dates
        df_train, df_test = self.split_data_by_date(df_ml_input_updated, cutoff_days)

        # Extract features and targets
        X_train, X_test, y_train, y_test = self.split_train_test(df_train, df_test, model_features, model_target)

        # Baseline model
        y_pred_baseline = self.baseline_model_predictions(df_train, y_test)
        mae_baseline, rmse_baseline = self.evaluate_model(y_test, y_pred_baseline)
        
        # XGBoost model
        xgb_model = self.build_xgboost_model(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        mae_xgb, rmse_xgb = self.evaluate_model(y_test, y_pred_xgb)
        self.save_model(xgb_model, "xgb_model")

        # LightGBM model
        lgb_model = self.build_lightgbm_model(X_train, y_train)
        y_pred_lgb = lgb_model.predict(X_test)
        mae_lgb, rmse_lgb = self.evaluate_model(y_test, y_pred_lgb)
        self.save_model(lgb_model, "lgb_model")

        return mae_baseline, rmse_baseline, mae_xgb, rmse_xgb, mae_lgb, rmse_lgb
    
    
if __name__ == "__main__":
    model_trainer = MLModelTrainer()
    mae_baseline, rmse_baseline, mae_xgb, rmse_xgb, mae_lgb, rmse_lgb = model_trainer.run()
    
    print(f"BASELINE. MAE: {mae_baseline} RMSE: {rmse_baseline}")
    print(f"XGBOOST. MAE: {mae_xgb} RMSE: {rmse_xgb}")
    print(f"LIGHTGBM. MAE: {mae_lgb} RMSE: {rmse_lgb}")