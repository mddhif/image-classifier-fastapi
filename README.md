# Image Classification API

This Python application is a FastAPI-based web service for image classification using the MobileNetV2 model pre-trained on ImageNet.

## Features

- Accepts image uploads via an API endpoint (`/predict/`).
- Preprocesses the uploaded image to match the model's input format.
- Uses MobileNetV2 to classify the image and predict the top three labels.
- Returns a JSON response with the predicted labels and their probabilities.

## How It Works

1. The API receives an image file.
2. The image is resized and preprocessed.
3. The model makes predictions.
4. The API responds with the top three predictions and their probabilities.

This app can be used for quick image classification tasks and can be extended for other deep-learning-based image recognition applications.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/mddhif/image-classification-api.git
   cd image-classification-api
