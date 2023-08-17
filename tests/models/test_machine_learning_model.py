import numpy as np
from unittest import TestCase

from app.models.machine_learning_model import MachineLearningModel


class TestMachineLearningModel(TestCase):
    def setUp(self) -> None:
        self.service = MachineLearningModel()

    def test_add_data_to_model(self):
        sample_data = [[1], [2], [3]]
        self.service.add_data_to_model(sample_data)
        self.assertEqual(self.service.data, sample_data)

        sample_data = []
        self.service.add_data_to_model(sample_data)
        self.assertEqual(self.service.data, sample_data)

    def test_find_representation_for_each_object(self):
        self.service.data = np.random.random((100, 2)) * 100
        self.service.find_representation_for_each_object()
        self.assertEqual(len(self.service.representation_values), 100)

    def test_train_model(self):
        self.service.data = np.random.random((100, 2)) * 100
        self.service.find_representation_for_each_object()
        self.service.train_model()
        self.assertEqual(self.service.model.n_features_in_, 2)