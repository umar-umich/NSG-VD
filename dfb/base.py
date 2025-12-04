import torch
import torch.nn as nn

class AbstractDetector(nn.Module):
    """
    All deepfake detectors should subclass this class.
    """
    def __init__(self, config=None):
        """
        config:   (dict)
            configurations for the model
        """
        super().__init__()

    def load_model(self, config=None):
        """
        Returns the features from the backbone given the input data.
        """
        pass

    def set_input(self, config=None):
        """
        Forward pass through the model, returning the prediction dictionary.
        """
        pass

    def get_predictions(self, config=None):
        """
        Builds the backbone of the model.
        """
        pass

    def generate_outputs(self, config=None):
        """
        Builds the backbone of the model.
        """
        pass
