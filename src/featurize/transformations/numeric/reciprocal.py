from ..base import BaseTransformer
import numpy as np


class ReciprocalTransformer(BaseTransformer):
    """
    Transformer that returns the reciprocal of a number
    """

    def __init__(self):
        self.name = "RECIPROCAL"

    def __call__(self, x):
        return np.select([x != 0, x == 0], [1 / x, 0])
