from ..base import BaseTransformer
import numpy as np


class CubeTransformer(BaseTransformer):
    """
    Transformer that returns the cube of a number
    """

    def __init__(self):
        self.name = "CUBE"

    def __call__(self, x):
        return x**3

    def get_description(self):
        return "Calculates the cube of a number."
