import datetime
import random
import numpy as np
import threading

from concurrent.futures import ThreadPoolExecutor

from app.models.machine_learning_model import MachineLearningModel


class MachineLearningService:
    """Service Implementation. Service is responsible for splitting collection of objects and train it
    concurrently."""
    def __init__(self):
        self.status = "Model not fitted"
        self.chunks_number: int = 5
        self.workers_number: int = self.chunks_number
        self._models = list()
        self._start_time = 0
        self._end_time = 0

    def train(self, data: np.array):
        """Method used to start training process"""
        self._start_time = datetime.datetime.now()
        self.status = f"Training lasts from: {self._start_time}"
        self._init_models()
        train_task = threading.Thread(target=self._train_multiple_models, kwargs={"data": data})
        train_task.start()

    def predict(self, data: np.array) -> list[np.array]:
        """Method used to predict."""
        predictions = list()
        for object_ in data:
            single_object_predictions = list()
            for model in self._models:
                prediction = model.predict(object_)
                if prediction:
                    single_object_predictions.append(prediction)
            if single_object_predictions:
                mean_prediction = np.mean(single_object_predictions)
                predictions.append(mean_prediction)

        return predictions

    def get_status(self) -> str:
        """Method used to get status"""
        return self.status

    def _init_models(self):
        """Method used to create machine learning models."""
        self._models: list = [MachineLearningModel() for _ in range(self.workers_number)]

    def _train_multiple_models(self, data: list[list[float]]):
        """Method used to train multiple models"""
        splitted_data: np.array = self._split_representation_collection(data)
        with ThreadPoolExecutor(max_workers=self.workers_number) as worker:
            for model, data_chunk in zip(self._models, splitted_data):
                worker.submit(self._train_single_model, model, data_chunk)
        self._end_time = datetime.datetime.now()
        self.status = f"Training lasted from: {self._start_time} to {self._end_time}"

    def _train_single_model(self, model: MachineLearningModel, representation_list: list):
        """Method used to train single model"""
        model.add_data_to_model(representation_list)
        try:
            model.find_representation_for_each_object()
            model.train_model()
        except Exception as e:
            self.status = (f"Error: {e} occurred: {datetime.datetime.now()}. "
                           f"Model started training at {self._start_time}")

    def _split_representation_collection(self, representation_list: list) -> list[list[float]]:
        """Method used to split collection of objects representations."""
        random.shuffle(representation_list)
        chunks = np.array_split(np.array(representation_list), self.chunks_number)
        return chunks
