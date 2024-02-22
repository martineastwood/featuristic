from ..base import BaseTransformer
import numpy as np


class ExpTransformer(BaseTransformer):
    """
    Transformer that returns the exponential of a number
    """

    def __init__(self):
        self.name = "EXP"

    def __call__(self, x):
        return np.exp(x)

    def get_description(self):
        return "Calculates the exponential of a number."
