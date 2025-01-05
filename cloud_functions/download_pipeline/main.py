import functions_framework
import yt_dlp
import os
from google.cloud import storage
from tempfile import NamedTemporaryFile
import cv2
import pandas_gbq
import time
import logging

IMG_DIR = "assets/images/"
TARGET_SIZE = (256, 256)

GCP_PROJECT_ID = "ascendant-cache-446408-v8"
GCP_VIDEO_URL_TABLE = "ascendant-cache-446408-v8.video_processing.video_urls"

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
    try:
        ydl_opts['logger'] = MyLogger()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(urls)
    except Exception as e:
        print(f"Error in yt_dlp_download: {str(e)}")
        raise


def download_videos(urls: list, path, title):
    yt_opts = {
        'format': 'worst',
        'outtmpl': path + title + '.%(ext)s',
        'ratelimit': 5000000,
        'cookiefile': 'cookies.txt'
    }

    yt_dlp_download(urls, yt_opts)


def upload_to_gcs(bucket_name, source_folder, destination_folder=None):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for root, _, files in os.walk(source_folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            if not file_name.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
                print(f"Skipping non-image file: {file_name}")
                continue

            relative_path = os.path.relpath(file_path, source_folder)
            blob_name = os.path.join(destination_folder, relative_path) if destination_folder else relative_path
            blob_name = blob_name.replace("\\", "/")  # Ensure GCS uses forward slashes

            blob = bucket.blob(blob_name)
            blob.upload_from_filename(file_path)
            print(f"Uploaded {file_path} to gs://{bucket_name}/{blob_name}")


def folder_exists_and_not_empty(bucket_name, folder_name):
    # Ensure folder_name ends with a trailing slash
    if not folder_name.endswith("/"):
        folder_name += "/"

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(client.list_blobs(bucket, prefix=folder_name))
    return len(blobs) > 0


def download_process_pipeline():
    df = pandas_gbq.read_gbq(f"SELECT * FROM `{GCP_VIDEO_URL_TABLE}`", project_id=GCP_PROJECT_ID)
    non_processed = df[df["status"] == "unprocessed"]

    for id, row in non_processed.iterrows():
        title = ''.join(e for e in row['title'] if e.isalnum())
        if not os.path.exists(title + '.mp4'):
            download_videos(row['url'], "", title)
            time.sleep(0.5)
        if process_vid(title + ".mp4", "test_video_images", title=row['title']):
            print("Processed:", title)
            df.loc[id, 'status'] = "processed"
            pandas_gbq.to_gbq(df, "video_processing.video_urls", "ascendant-cache-446408-v8", if_exists='replace')
            print("Updated csv file.")
        else:
            print("Could not process:", title)
            df.loc[id, 'status'] = "failed"
        time.sleep(5)
    pandas_gbq.to_gbq(df, "video_processing.video_urls", "ascendant-cache-446408-v8", if_exists='replace')
    print("Updated csv file.")


def process_vid(video_name, bucket_name, title):
    print(f"Bucket name:", bucket_name)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    vidcap = cv2.VideoCapture(video_name)
    count = 0
    success = True
    images_created = False
    print("video name:", video_name)
    if vidcap.isOpened():
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        print("fps:", fps)
        if fps == 0:
            fps = 25

        folder_name = f"images/{title}"
        if folder_exists_and_not_empty(bucket_name, folder_name):
            print("image directory already exists")  # TODO: better check to see if video has been processed
            return True
        while success:
            success, image = vidcap.read()
            if count % (1 * fps) == 0:  # capture image at each second
                image = cv2.resize(image, TARGET_SIZE)
                with NamedTemporaryFile() as temp:
                    # Extract name to the temp file
                    iName = "".join([str(temp.name), ".jpg"])

                    # Save image to temp file
                    cv2.imwrite(iName, image)

                    blob_name = f"images/{title}/sec_{int(count / fps)}_.jpg"
                    print("blob name:", blob_name)

                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(iName, content_type='image/jpeg')
                images_created = True
            count += 1

    return images_created


# Triggered from a message on a Cloud Pub/Sub topic.
@functions_framework.cloud_event
def hello_pubsub(cloud_event):
    print("Started download pipeline")
    download_process_pipeline()
    print("Finished download pipeline task")
    return 200
