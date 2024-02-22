from ..base import BaseTransformer
import numpy as np


class CosineTransformer(BaseTransformer):
    """
    Transformer that returns the cosine of a number
    """

    def __init__(self):
        self.name = "COSINE"

    def __call__(self, x):
        return np.cos(x)

    def get_description(self):
        return "Calculates the cosine of a number."
