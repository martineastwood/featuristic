# This file contains the base class for all transformers


class BaseTransformer:
    """
    Base class for all transformers
    """

    def __init__(self):
        self.name = "UNDEFINED"

    def __call__(self, x):
        """
        Applies the transformation to the input

        args:
            x: any input to be transformed
        """
        raise NotImplementedError

    def get_column_names(self, col_name):
        """
        Returns the name of the column after transformation

        args:
        col_name: str
            name of the column before transformation
        """
        return f"{self.name}({col_name})"

    def get_description(self):
        raise NotImplementedError


class BaseCombinationTransformer:
    """
    Base class for all combination transformers
    """

    def __init__(self):
        self.name = "UNDEFINED"

    def __call__(self, x, y):
        """
        Applies the transformation to the input

        args:
            x: any input to be transformed
            y: any input to be transformed
        """
        raise NotImplementedError

    def get_column_names(self, col_name1, col_name2):
        """
        Returns the name of the column after transformation

        args:
        col_name1: str
            name of the column before transformation
        col_name2: str
            name of the column before transformation
        """
        return f"{self.name}({col_name1}, {col_name2})"

    def get_description(self):
        raise NotImplementedError
