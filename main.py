from typing import Optional

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import urllib.request
import numpy as np
import torch
import utils

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
    "https://facial-emotion-recognition-react.vercel.app",
    "https://facial-emotion-recognition-react-git-main-zulumkhuzo-gmailcom.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class_names = ['anger',
               'contempt',
               'disgust',
               'fear',
               'happiness',
               'neutral',
               'sadness',
               'surprise']

# class_names = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

loaded_model = utils.load_model(model_path="models/fer_plus_best_model_latest.pth", class_names=class_names, device="cpu")


class Feature(BaseModel):
    photo: Optional[str]


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        data = await websocket.receive_text()
        await websocket.send_text(data)


@app.post("/predict")
async def predict_emotion(feature: Feature):
    emotion = ""
    with urllib.request.urlopen(feature.photo) as req:
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    full_size_image = cv2.imdecode(arr, -1)
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face.detectMultiScale(full_size_image, 1.3, 10)

    for (x, y, w, h) in faces:
        roi = full_size_image[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # predicting the emotion
        img = torch.tensor(cropped_img)
        img_input = torch.permute(img.squeeze(), (2, 0, 1)).unsqueeze(0)
        X = img_input.type(torch.float) / 255.0
        with torch.inference_mode():
            Y_logit = loaded_model(X)
        Y_pred = torch.argmax(Y_logit.squeeze(), dim=0)
        emotion = class_names[int(Y_pred)]
        print("Emotion: " + class_names[int(Y_pred)])

    return {
        "emotion": emotion
    }
