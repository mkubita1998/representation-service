import numpy as np
from fastapi import FastAPI

from app.services import ml_service

app = FastAPI()


@app.post("/train")
def train(data: list[list[float]]):
    ml_service.train(np.array(data))
    return {"message": "training started"}


@app.post("/predict")
def predict(data: list[list[float]]):
    predictions = ml_service.predict(np.array(data))
    return {"predictions": predictions}


@app.get("/status")
def get_status():
    status = ml_service.get_status()
    return {"status": status}
