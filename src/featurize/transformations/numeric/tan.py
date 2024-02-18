from ..base import BaseTransformer
import numpy as np


class TanTransformer(BaseTransformer):
    """
    Transformer that returns the tangent of a number
    """

    def __init__(self):
        self.name = "TAN"

    def __call__(self, x):
        return np.tan(x)
