import numpy as np
from unittest import TestCase
from unittest.mock import patch

from app.services.machine_learning_service import MachineLearningService


class TestMachineLearningService(TestCase):
    def setUp(self) -> None:
        self.service = MachineLearningService()

    def test_get_status(self):
        self.assertEqual(self.service.get_status(), "Model not fitted")

    def test_train(self):
        data = np.random.random((100, 2)) * 100
        with patch("threading.Thread") as thread_:
            self.service.train(data)
            thread_.assert_called()
        self.assertIn("Training lasts from:", self.service.status)

    def test_predict(self):
        data = np.random.random((10, 2)) * 100
        self.assertListEqual(self.service.predict(data), list())
