from ..base import BaseTransformer
import numpy as np


class AbsoluteTransformer(BaseTransformer):
    """
    Transformer that returns the absolute value of a number
    """

    def __init__(self):
        self.name = "ABS"

    def __call__(self, x):
        return np.abs(x)

    def get_description(self):
        return "Calculates the absolute value of a number."
