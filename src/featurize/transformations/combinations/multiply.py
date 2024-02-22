from ..base import BaseCombinationTransformer


class MultiplyTransformer(BaseCombinationTransformer):
    """
    Transformer that returns the multiplication of two numbers
    """

    def __init__(self):
        self.name = "MULTIPLICATION"

    def __call__(self, x, y):
        return x * y

    def get_description(self):
        return "Calculates the multiplication of two numbers."
