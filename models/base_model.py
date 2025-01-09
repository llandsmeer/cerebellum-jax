# models/base_model.py

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    A base interface for RL models. Subclass this and implement the needed methods.
    """

    @abstractmethod
    def predict(self, obs):
        """
        Given an observation, return the model's action.
        """
        pass

    @abstractmethod
    def train_step(self, batch):
        """
        Perform a single training step using a batch of experience.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path):
        """
        Save model parameters to the specified path.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path):
        """
        Load model parameters from the specified path.
        """
        pass
