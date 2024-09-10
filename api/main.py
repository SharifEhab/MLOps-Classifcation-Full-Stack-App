## This is the main file where we will write our FastAPI code.
## We will use FastAPI to create a REST API that will accept an image file and return the prediction of the model.
## FastAPI as backend and uvicorn as ASGI server.

from fastapi import FastAPI, File, UploadFile
import uvicorn # ASGI server (Asynchronous Server Gateway Interface) 
import numpy as np
from io import BytesIO
from PIL import Image

import tensorflow as tf

app = FastAPI()


MODEL = tf.keras.models.load_model('./saved_models/1') 
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
    prediction = CLASS_NAME[np.argmax(MODEL.predict(img_batch)[0])]
    confidence = np.max(MODEL.predict(img_batch)[0])
    return {
        'prediction': prediction
        , 'confidence': float(confidence)
    }



if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
