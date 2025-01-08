import gradio as gr
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import cv2
from keras.models import load_model
from keras.models import Model
from datasets import load_dataset
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from PIL import Image

model_path = hf_hub_download(repo_id="eybro/autoencoder", filename="autoencoder_model.keras", repo_type='model')
data_path = hf_hub_download(repo_id="eybro/encoded_images", filename="X_encoded_compressed.npy", repo_type='dataset')

autoencoder = load_model(model_path)
encoded_images = np.load(data_path)

dataset = load_dataset("eybro/images")
split_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)  # 80% train, 20% test
dataset['train'] = split_dataset['train']
dataset['test'] = split_dataset['test']

example_images = {
    "Example 1": "example_1.png",
    "Example 2": "example_2.png",
    "Example 3": "example_3.jpg"
}

def create_url_from_title(title: str, timestamp: int):
    video_urls = load_dataset("eybro/video_urls")
    df = video_urls['train'].to_pandas()
    filtered = df[df['title'] == title]
    base_url = filtered.iloc[0, :]["url"]
    return base_url + f"&t={timestamp}s"

def find_nearest_neighbors(encoded_images, input_image, top_n=5):
    """
    Find the closest neighbors to the input image in the encoded image space.
    Args:
    encoded_images (np.ndarray): Array of encoded images (shape: (n_samples, n_features)).
    input_image (np.ndarray): The encoded input image (shape: (1, n_features)).
    top_n (int): The number of nearest neighbors to return.
    Returns:
    List of tuples: (index, distance) of the top_n nearest neighbors.
    """
    # Compute pairwise distances
    distances = euclidean_distances(encoded_images, input_image.reshape(1, -1)).flatten()

    # Sort by distance
    nearest_neighbors = np.argsort(distances)[:top_n]
    return [(index, distances[index]) for index in nearest_neighbors]

def get_image(index):
    split = len(dataset["train"])
    if index < split:
        return dataset["train"][index]
    else:
        return dataset["test"][index-split]

def process_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, (64, 64))  
    img = img.astype('float32')  
    img /= 255.0
    img = np.expand_dims(img, axis=0)

    layer_model = Model(inputs=autoencoder.input, outputs=autoencoder.layers[4].output)

    encoded_array = layer_model.predict(img) 

    pooled_array = encoded_array.max(axis=-1)
    return pooled_array  # Shape: (1, n_features)
    
def inference(user_image=None, selected_example=None):

    if user_image is not None and selected_example is not None:
        return "Please upload an image or select an example image."
    elif user_image is not None:
        input_image = process_image(user_image)
    elif selected_example is not None:
        input_image = load_example(selected_example)
        input_image = process_image(input_image)
    else:
        return "Please upload an image or select an example image."

    nearest_neighbors = find_nearest_neighbors(encoded_images, input_image, top_n=5)
    
    top4 = [int(i[0]) for i in nearest_neighbors[:4]]
    
    for i in top4:
      im = get_image(i)
      print(im["label"], im["timestamp"])

    result_image = get_image(top4[0])
    url = create_url_from_title(result_image['label'], result_image['timestamp'])
    result = f"{result_image['label']} {result_image['timestamp']} \n{url}"
    
    return result

def load_example(example_name):
    image_path = example_images.get(example_name)
    if image_path:
        return Image.open(image_path)
    return None
           
with gr.Blocks() as demo:
    gr.Markdown("""
        # Image to Video App
        Find your favorite Gordon Ramasay scene by uploading an image from the scene, the app will thereafter find a corresponding youtube video for that scene. 
        Or try one of our examples - Screenshots form Youtube videos.
        """)

    with gr.Row():
        with gr.Column():
            inp_image = gr.Image(label="Upload Image", type="pil")
            example_selection = gr.Radio(
                choices=list(example_images.keys()),
                label="Select Example Image",
                type="value"  # Ensure single string return value
            )
            example_display = gr.Image(label="Selected Example Image", type="pil")

        with gr.Column():
            output = gr.Markdown()

    
    example_selection.change(
        lambda selected_example: load_example(selected_example),
        inputs=[example_selection],
        outputs=[example_display]
    )
    
    clear_button = gr.Button("Clear Example")
    
    clear_button.click(
        lambda: (None, None), 
        inputs=[],
        outputs=[example_selection, example_display]
    )

    submit_button = gr.Button("Submit")

    submit_button.click(
        lambda user_image, selected_example: inference(user_image=user_image, selected_example=selected_example),
        inputs=[inp_image, example_selection],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()