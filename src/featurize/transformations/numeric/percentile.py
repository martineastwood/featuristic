from ..base import BaseTransformer
import numpy as np


class PercentileTransformer(BaseTransformer):
    """
    Transformer that returns the percentile of a number
    """

    def __init__(self):
        self.name = "PERCENTILE"

    def __call__(self, x):
        return x.rank(pct=True)
