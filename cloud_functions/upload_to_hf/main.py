import os
from google.cloud import storage
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from io import BytesIO
from datasets import Dataset
from google.colab import userdata
from huggingface_hub import HfApi
from huggingface_hub import login
from datasets import load_dataset
from tqdm import tqdm


def fetch_existing_video_labels():
      dataset = load_dataset("eybro/images", split="train")
      existing_labels = set(dataset["label"])
      return existing_labels

def fetch_images_from_gcs(bucket):
    data = []

    for blob in bucket.list_blobs(prefix="images/"):
        if blob.name.endswith(('.jpg', '.png')):
            parts = blob.name.split('/')
            video_name = parts[1]

            filename = parts[2]
            timestamp = int(filename.split('_')[1].split('.')[0])

            image_data = blob.download_as_bytes()
            image = Image.open(BytesIO(image_data)).convert("RGB")

            data.append({
                'image': image,
                'label': video_name,
                'timestamp': timestamp
            })

    return data

def fetch_images_from_gcs(bucket, existing_labels):
    data = []

    total_images = sum(1 for _ in bucket.list_blobs(prefix="images/"))
    for blob in tqdm(bucket.list_blobs(prefix="images/"), total=total_images, desc="Processing images", unit="image"):

      if blob.name.endswith(('.jpg', '.png')):
          parts = blob.name.split('/')
          video_name = parts[1]

          # Skip this video if it's already in the dataset
          if video_name in existing_labels:
            continue

          filename = parts[2]
          timestamp = int(filename.split('_')[1].split('.')[0])

          image_data = blob.download_as_bytes()
          image = Image.open(BytesIO(image_data)).convert("RGB")

          data.append({
              'image': image,
              'label': video_name,
              'timestamp': timestamp
          })

    return data


def push_to_hub(data_batch):
    dataset = Dataset.from_dict({
        'image': [item['image'] for item in data_batch],
        'label': [item['label'] for item in data_batch],
        'timestamp': [item['timestamp'] for item in data_batch],
    })
    dataset.push_to_hub("eybro/images")

push_to_hub(data)

def main():
    token = os.environ["hf_token"]
    login(token=token)
    client = storage.Client()
    bucket_name = "test_video_images"
    bucket = client.bucket(bucket_name)

    existing_labels = fetch_existing_video_labels()

    data = fetch_images_from_gcs(bucket, existing_labels)
    push_to_hub(data)


@functions_framework.cloud_event
def hello_pubsub(cloud_event):
    main()
