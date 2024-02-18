from ..base import BaseCombinationTransformer


class DivideTransformer(BaseCombinationTransformer):
    """
    Transformer that returns the division of two numbers
    """

    def __init__(self):
        self.name = "DIVIDE"

    def __call__(self, x, y):
        return x / y
