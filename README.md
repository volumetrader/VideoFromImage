# Image to Youtube Video Project
This project is a complete ML system designed to help users identify the specific YouTube video and timestamp that correspond to their uploaded image. 
Our data consists of images scraped of various vides from Gordon Ramsay TV shows and the service is accessible [here.](https://huggingface.co/spaces/eybro/image_video_timestamp)

# Overview

The system can be splitted into two parts:
* Data collection: Scheduled scraping of Youtube videos to extract images from each timestamp
* End-to-End Prediction System: Feature extraction, model building and inference.

# **Data Collection**
![Architecture of Data collection](report_images/Data.png)

Firstly a few youtube playlist were chosen as a dataset to get a decent amount of data 
but not too much for us to process. We went with a playlist from the youtube channel "Kithchen Nightmates" with
Gordon Ramsey.

The entire data collection pipeline was set up in google cloud, it consists of three cloud
functions and one database table.

The first function takes the playlist urls scraped data from youtube about their contents
and inserts into a database table (if video ids are not already present)

The second function goes through this table and downloads videos which are not alread
marked as processed, then splits the video into images at one second increments. The resulting
images are stored in a storage bucket on google cloud.

Lastly a script takes all the images and creates a labeled dataset which is uploaded to huggingface.


# **End-to-End Prediction System**
![Architecture of ML system](report_images/ML.png)
