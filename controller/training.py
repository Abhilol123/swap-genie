import os

from dao.repository.training import Predictor


class Training:
    def __init__(self):
        self.predictor = Predictor()

    def train(self):
        model_path = self.predictor.predict(
            instance_data="",
            task="face",
            seed=42,
            resolution=512
        )
        return model_path
