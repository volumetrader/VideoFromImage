import functions_framework
import os
from google.cloud import storage
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from io import BytesIO
from datasets import Dataset


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

def main():
    client = storage.Client()
    bucket_name = "test_video_images"
    bucket = client.bucket(bucket_name)

    data = fetch_images_from_gcs(bucket)

    dataset = Dataset.from_dict({
        'image': [item['image'] for item in data],
        'label': [item['label'] for item in data],
        'timestamp': [item['timestamp'] for item in data],
    })

    print(dataset[0])
    print(len(dataset))

    token = os.environ["hf_token"]
    login(token=token)
    dataset.push_to_hub("eybro/images")


@functions_framework.cloud_event
def hello_pubsub(cloud_event):
    main()
