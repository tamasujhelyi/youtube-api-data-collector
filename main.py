import os
from typing import Dict, List

from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
import pandas as pd
import pandas_gbq


class YouTubeVideoDataCollector:
    def __init__(self, num_videos: int):
        """Initialize the YouTube video data collector.
        
        Args:
            num_videos: Number of videos to collect data for.
        """

        # Load environment variables
        load_dotenv()
        
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        self.youtube_channel_id = os.getenv("YOUTUBE_CHANNEL_ID")
        self.bigquery_project_id = os.getenv("BIGQUERY_PROJECT_ID")
        self.bigquery_table_id = os.getenv("BIGQUERY_TABLE")
        # Number of videos to collect data for
        self.num_videos = num_videos
        
        # Check for presence of required environment variables
        if not all([self.youtube_api_key, self.youtube_channel_id, self.bigquery_project_id, self.bigquery_table_id]):
            raise ValueError("Missing required environment variables.")
            
    def fetch_uploads_playlist_id(self) -> str:
        """Fetch the uploads playlist id (playlist containing all uploaded videos on a YouTube channel)."""

        with build("youtube", "v3", developerKey=self.youtube_api_key) as youtube_service:
            channels_response = youtube_service.channels().list(
                part="contentDetails",
                id=self.youtube_channel_id
            ).execute()

            uploads_playlist_id = channels_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

            return uploads_playlist_id
    
    def fetch_video_info(self, uploads_playlist_id) -> List[Dict]:
        """Fetch basic infos (title, publish datetime) about videos from the channel's uploads playlist."""

        video_info = []
        next_page_token = None
        
        with build("youtube", "v3", developerKey=self.youtube_api_key) as youtube_service:
            while len(video_info) < self.num_videos:
                playlist_items_response = youtube_service.playlistItems().list(
                    part="snippet",
                    playlistId=uploads_playlist_id,
                    maxResults=min(50, self.num_videos - len(video_info)),
                    pageToken=next_page_token
                ).execute()
                
                for item in playlist_items_response["items"]:
                    video_info.append({
                        "video_id": item["snippet"]["resourceId"]["videoId"],
                        "video_title": item["snippet"]["title"],
                        "video_publish_datetime": item["snippet"]["publishedAt"]
                    })
                
                next_page_token = playlist_items_response.get("nextPageToken")
                
                if not next_page_token:
                    break
                    
        return video_info
    
    def fetch_video_statistics(self, video_ids: List[str]) -> List[Dict]:
        """Fetch statistics (duration, view count) for a list of video IDs."""

        video_statistics = []
        batch_size = 50
        
        with build("youtube", "v3", developerKey=self.youtube_api_key) as youtube_service:
            for i in range(0, len(video_ids), batch_size):
                batch = video_ids[i:i + batch_size]
                videos_response = youtube_service.videos().list(
                    part="contentDetails,statistics",
                    id=",".join(batch)
                ).execute()
                
                for video in videos_response["items"]:
                    video_statistics.append({
                        "video_id": video["id"],
                        "video_duration": video["contentDetails"]["duration"],
                        "video_view_count": video["statistics"]["viewCount"]
                    })
                    
        return video_statistics
    
    def process_and_combine_video_data(self, video_info: List[Dict], video_statistics: List[Dict]) -> pd.DataFrame:
        """Process and combine video info and statistics into a single DataFrame."""

        df_video_info = pd.DataFrame(video_info)
        df_video_statistics = pd.DataFrame(video_statistics)
        
        df_video_all_data = (
            df_video_info.merge(df_video_statistics, how="inner", on="video_id")
            .assign(
                video_publish_datetime=lambda df_: pd.to_datetime(df_["video_publish_datetime"]),
                video_view_count=lambda df_: df_["video_view_count"].astype("int")
            )
        )
        
        return df_video_all_data
    
    def load_to_bigquery(self, df_video_all_data: pd.DataFrame) -> None:
        """Load the processed DataFrame to BigQuery."""

        credentials = service_account.Credentials.from_service_account_file(
            "service-account-key.json"
        )
        
        pandas_gbq.to_gbq(
            df_video_all_data,
            self.bigquery_table_id,
            project_id=self.bigquery_project_id,
            if_exists="replace",
            credentials=credentials
        )
    
    def run(self) -> pd.DataFrame:
        """Execute the full data collection and loading pipeline."""

        uploads_playlist_id = self.fetch_uploads_playlist_id()
        video_info = self.fetch_video_info(uploads_playlist_id)
        video_ids = [video["video_id"] for video in video_info]
        video_statistics = self.fetch_video_statistics(video_ids)
        
        df_video_all_data = self.process_and_combine_video_data(video_info, video_statistics)
        self.load_to_bigquery(df_video_all_data)
        
        return df_video_all_data


if __name__ == "__main__":
    collector = YouTubeVideoDataCollector(num_videos=3000)
    df_video_all_data  = collector.run()
    print(df_video_all_data.head(10))