
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions, MobileNetV2
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image



app = FastAPI()

print("Loading the model ...")
model = MobileNetV2(weights="imagenet")

def read_image(file) -> Image.Image:
    print("Preprocessing and preparing the image")
    img = Image.open(BytesIO(file)).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    print("Reading uploaded image ...")
    img_array = read_image(await file.read())
    predictions = model.predict(img_array) 
    decoded_predictions = decode_predictions(predictions, top=3)[0]  

    results = [{"label": label, "probability": float(prob)} for (_, label, prob) in decoded_predictions]
    print("Sending results back ...")
    return {"predictions": results}


