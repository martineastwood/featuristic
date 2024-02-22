from ..base import BaseTransformer
import numpy as np


class SineTransformer(BaseTransformer):
    """
    Transformer that returns the sine of a number
    """

    def __init__(self):
        self.name = "SINE"

    def __call__(self, x):
        return np.sin(x)

    def get_description(self):
        return "Calculates the sine of a number."
