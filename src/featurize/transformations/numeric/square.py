from ..base import BaseTransformer
import numpy as np


class SquareTransformer(BaseTransformer):
    """
    Transformer that returns the square of a number
    """

    def __init__(self):
        self.name = "SQUARE"

    def __call__(self, x):
        return x**2
