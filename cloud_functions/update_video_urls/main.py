import functions_framework
import yt_dlp
import pandas as pd
import pandas_gbq
import logging

GCP_PROJECT_ID = "ascendant-cache-446408-v8"
GCP_VIDEO_URL_TABLE = "ascendant-cache-446408-v8.video_processing.video_urls"
images_path = 'assets/videos/train'
URL_DF_COLUMNS = ["url", "title", "id", "duration", "channel", "channel_id", "status"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MyLogger:
    def debug(self, msg):
        if msg.startswith('[debug] '):
            logger.debug(msg)
        else:
            self.info(msg)

    def info(self, msg):
        logger.info(msg)

    def warning(self, msg):
        logger.warning(msg)

    def error(self, msg):
        logger.error(msg)


def yt_dlp_download(urls: list, ydl_opts: dict):
    """Download media using yt-dlp and save it to a local file."""
    video_urls = []
    try:
        ydl_opts['logger'] = MyLogger()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for playlist_url in urls:
                try:
                    # Extract info
                    info_dict = ydl.extract_info(playlist_url, download=False)
                    if 'entries' in info_dict:
                        for entry in info_dict['entries']:
                            entry["playlist_url"] = playlist_url
                            video_urls.append(entry)
                    else:
                        print("The provided URL is not a playlist.")
                except Exception as e:
                    print(f"An error occurred: {e}")
    except Exception as e:
        print(f"Error in yt_dlp_download: {str(e)}")
        raise
    return video_urls


def get_playlist_urls(urls: list):
    options = {
        'quiet': True,
        'extract_flat': True
    }

    video_urls = yt_dlp_download(urls, options)

    df = pd.DataFrame(video_urls)
    df = df.loc[:, ["url", "title", "id", "duration", "channel", "channel_id", "playlist_url"]]
    return df


def update_urls_file(playlist_urls: list):
    new_df = get_playlist_urls(playlist_urls)
    old_df = pandas_gbq.read_gbq(f"SELECT * FROM `{GCP_VIDEO_URL_TABLE}`", project_id=GCP_PROJECT_ID)

    new_ids = [i for i in new_df["id"].to_numpy() if i not in list(old_df["id"])]
    new_df = new_df.set_index("id")
    old_df = old_df.set_index("id")
    new_df["status"] = "unprocessed"
    new_df["title"] = new_df["title"].apply(lambda x: " ".join(x.splitlines()))

    df = pd.concat([old_df, new_df.loc[new_ids, :]])
    pandas_gbq.to_gbq(df, GCP_VIDEO_URL_TABLE, GCP_PROJECT_ID, if_exists='replace')
    print(f"Added {len(new_ids)} new videos to the table.")


def main():
    playlist_urls = ["https://www.youtube.com/playlist?list=PLoieWSdjTgUL3uVRfTgbWa-n7sw9vMunZ",
                     "https://www.youtube.com/playlist?list=PL69457A7D9A55C71F"]
    update_urls_file(playlist_urls)


@functions_framework.cloud_event
def hello_pubsub(cloud_event):
    main()
    return 200