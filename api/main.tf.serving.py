## Use different approach to serve the model using tensorflow serving
## We will use tensorflow serving to serve the model and FastAPI to create a REST API that will accept an image file and return the prediction of the model

from fastapi import FastAPI, File, UploadFile
import uvicorn # ASGI server
import numpy as np
from io import BytesIO
from PIL import Image

import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()


origins = [
    "http://localhost", 
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


endpoint = "http://localhost:8501/v1/models/potatoes_model:predict" # use latest version of the model for prediction, you can also use specific version
CLASS_NAME = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "PONG!!!"



def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

#fastapi provides in built data validation
@app.post("/predict")
async def predict( 
  file: UploadFile = File(...),
):
    image = read_file_as_image(await file.read())

    img_batch = np.expand_dims(image, 0) # create a batch of 1 image since predict requires a batch
    json_data = {"instances": img_batch.tolist()}
    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()['predictions'][0])
    pridected_class = CLASS_NAME[np.argmax(prediction)]
    confidence = np.max(prediction)
    return {
        "class": pridected_class,
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
