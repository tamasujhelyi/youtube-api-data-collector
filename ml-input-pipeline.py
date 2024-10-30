from datetime import datetime
import os

from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
import isodate
import pandas as pd
import pandas_gbq
from textblob import TextBlob


class MLInputPipeline:
    """A class to handle YouTube data feature engineering and BigQuery operations."""

    def __init__(self):
        """Initialize the MLInputPipeline class."""

        # Load environment variables
        load_dotenv()

        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        self.youtube_channel_id = os.getenv("YOUTUBE_CHANNEL_ID")
        self.bigquery_project_id = os.getenv("BIGQUERY_PROJECT_ID")
        self.bigquery_table=os.getenv("BIGQUERY_TABLE")
        self.bigquery_ml_input_table = os.getenv("BIGQUERY_ML_INPUT_TABLE")
        self.credentials = service_account.Credentials.from_service_account_file(
            "service-account-key.json"
        )

        # Check for presence of required environment variables
        if not all([self.youtube_api_key, self.youtube_channel_id, self.bigquery_project_id, self.bigquery_table, self.bigquery_ml_input_table, self.credentials]):
            raise ValueError("Missing required environment variables.")

    def fetch_channel_creation_datetime(self) -> datetime:
        """Fetch channel creation datetime from YouTube API.

        Returns:
            datetime: Channel creation datetime
        """
        
        with build("youtube", "v3", developerKey=self.youtube_api_key) as youtube_service:
            channels_response = youtube_service.channels().list(
                part="snippet",
                id=self.youtube_channel_id
            ).execute()

            channel_creation_datetime_str = channels_response["items"][0]["snippet"]["publishedAt"]
            channel_creation_datetime = pd.to_datetime(channel_creation_datetime_str)

            return channel_creation_datetime
    
    def read_bigquery_table(self, sql_query: str) -> pd.DataFrame:
        """Read data from BigQuery table.
        
        Args:
            sql_query (str): SQL query to execute
            
        Returns:
            pd.DataFrame: Query results as DataFrame
        """

        df = pandas_gbq.read_gbq(
            sql_query,
            project_id=self.bigquery_project_id,
            credentials=self.credentials
        )

        return df

    def _map_hour_to_time_of_day(hour: int) -> int:
        """Map hour to time of day category.
    
        Args:
            hour (int): Hour of day (0-23)

        Returns:
            int: Time of day category (0: Morning, 1: Afternoon, 2: Evening, 3: Night)
        """

        if 5 <= hour <= 11:
            return 0 # Morning
        elif 12 <= hour <= 16:
            return 1 # Afternoon
        elif 17 <= hour <= 20:
            return 2 # Evening
        else:
            return 3 # Night
    
    def create_df_ml_input(self, df: pd.DataFrame, channel_creation_datetime: datetime) -> pd.DataFrame:
        """Create ML input DataFrame with engineered features.
        
        Args:
            df (pd.DataFrame): Raw data DataFrame
            channel_creation_datetime (datetime): Channel creation datetime
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """

        df = df.copy()
        
        df = (
            df.assign(
                duration_seconds=lambda df_: df_["video_duration"].apply(lambda x: isodate.parse_duration(x).total_seconds()).astype("int"),
                publish_year=lambda df_: df_["video_publish_datetime"].dt.year,
                publish_month=lambda df_: df_["video_publish_datetime"].dt.month,
                publish_quarter=lambda df_: df_["video_publish_datetime"].dt.quarter,
                publish_day=lambda df_: df_["video_publish_datetime"].dt.day,
                publish_day_of_week=lambda df_: df_["video_publish_datetime"].dt.day_of_week,
                publish_hour=lambda df_: df_["video_publish_datetime"].dt.hour,
                publish_is_weekend=lambda df_: df_["publish_day_of_week"].apply(lambda x: 1 if x >= 5 else 0),
                publish_time_of_day=lambda df_: df_["publish_hour"].apply(self._map_hour_to_time_of_day),
                days_since_channel_creation=lambda df_: df_["video_publish_datetime"].sub(channel_creation_datetime).dt.days,
                title_length=lambda df_: df_["video_title"].apply(lambda x: len(x)),
                title_word_count=lambda df_: df_["video_title"].apply(lambda x: len(str(x).split())),
                title_sentiment_score=lambda df_: df_["video_title"].apply(lambda x: round(TextBlob(x).sentiment.polarity, 2)),
            )
            .rename(columns={
                "video_id":"id",
                "video_title":"title",
                 "video_publish_datetime":"publish_datetime",
                 "video_view_count":"view_count"
            })
            .drop(columns=["video_duration"])
        )

        return df
    
    def load_to_bigquery(self, df: pd.DataFrame) -> None:
        """Load DataFrame to BigQuery table.
        
        Args:
            df (pd.DataFrame): DataFrame to load
        """

        pandas_gbq.to_gbq(
            df,
            self.bigquery_ml_input_table,
            project_id=self.bigquery_project_id,
            if_exists="replace",
            credentials=self.credentials
        )

    def run(self, sql_query: str) -> None:
        """Run the complete ML input pipeline.
        
        Args:
            sql_query (str): SQL query to fetch raw data
        """
        
        channel_creation_datetime = self.fetch_channel_creation_datetime()
        df_raw_data = self.read_bigquery_table(sql_query)
        df_ml_input = self.create_df_ml_input(df_raw_data, channel_creation_datetime)
        self.load_to_bigquery(df_ml_input)


if __name__ == "__main__":
    ml_input = MLInputPipeline()

    sql_query = f"""
    select
        video_id
        , video_title
        , video_publish_datetime
        , video_duration
        , video_view_count
    from
        `{ml_input.bigquery_project_id}.{ml_input.bigquery_table}`
    """

    ml_input.run(sql_query)