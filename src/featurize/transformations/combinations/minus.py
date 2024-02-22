from ..base import BaseCombinationTransformer


class SubtractTransformer(BaseCombinationTransformer):
    """
    Transformer that returns the difference of two numbers
    """

    def __init__(self):
        self.name = "SUBTRACT"

    def __call__(self, x, y):
        return x - y

    def get_description(self):
        return "Calculates the difference between two numbers."
