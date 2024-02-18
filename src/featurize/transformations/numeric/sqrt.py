from ..base import BaseTransformer
import numpy as np


class SqrtTransformer(BaseTransformer):
    """
    Transformer that returns the square root of a number
    """

    def __init__(self):
        self.name = "SQRT"

    def __call__(self, x):
        return np.sqrt(x)
