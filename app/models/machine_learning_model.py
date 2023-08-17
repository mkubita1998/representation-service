import logging

import numpy as np
import sklearn.exceptions

from scipy.spatial import KDTree
from sklearn.tree import DecisionTreeRegressor
from typing import Optional


class MachineLearningModel:
    def __init__(self):
        """Machine Learning Model implementation with all necessary algorithms."""
        self.model = DecisionTreeRegressor()
        self.data: list[list[float]] = list()
        self.k_number_nearest_neighbours = 3
        self.nearest_neighbours_distances: list[float] = list()
        self.representation_values: list[float] = list()

    def add_data_to_model(self, representation_list: list[list[float]]):
        """Method used to add data to model"""
        self.data = representation_list

    def find_representation_for_each_object(self):
        """Method used to find representation for each object."""
        for object_ in self.data:
            k_nearest_neighbours_distances = self._find_distance_to_k_nearest_neighbours_for_single_object(object_)
            rep_value = self._calculate_representation_value_for_single_object(k_nearest_neighbours_distances)
            self.representation_values.append(rep_value)

    def train_model(self):
        """Method used to train model."""
        x = self.data
        y = self.representation_values
        self.model.fit(x, y)

    def predict(self, data: np.array) -> Optional[float]:
        """Method used to normalize data for training and make predictions."""
        x = data
        try:
            return self.model.predict(x.reshape(1, -1))
        except (sklearn.exceptions.NotFittedError, np.exceptions.AxisError, ValueError) as e:
            logging.error(e)
            return None

    def _find_distance_to_k_nearest_neighbours_for_single_object(self, object_: list[float]) -> list[float]:
        """Method used to find distance to K nearest neighbours using KD tree algorithm."""
        distances: list[float] = list()
        temp_data = self.data
        for _ in range(self.k_number_nearest_neighbours):
            distance, nearest_neighbour_index = KDTree(temp_data).query(object_)
            distances.append(distance)
            temp_data = np.delete(temp_data, nearest_neighbour_index, axis=0)
        return distances

    @staticmethod
    def _calculate_representation_value_for_single_object(distances: list[float]) -> float:
        """Method used to calculate representation value for single object"""
        mean = np.mean(distances)
        representation_value = 1 / (1 + mean)
        return representation_value
