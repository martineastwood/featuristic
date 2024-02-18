from ..base import BaseCombinationTransformer


class PlusTransformer(BaseCombinationTransformer):
    """
    Transformer that returns the sum of two numbers
    """

    def __init__(self):
        self.name = "PLUS"

    def __call__(self, x, y):
        return x + y
