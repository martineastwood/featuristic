from .absolute import AbsoluteTransformer
from .cosine import CosineTransformer
from .cube import CubeTransformer
from .exp import ExpTransformer
from .sine import SineTransformer
from .sqrt import SqrtTransformer
from .square import SquareTransformer
from .tan import TanTransformer
from .reciprocal import ReciprocalTransformer


transfomers = {
    "ABSOLUTE": AbsoluteTransformer(),
    # "COSINE": CosineTransformer(),
    "CUBE": CubeTransformer(),
    # "EXP": ExpTransformer(),
    # "SINE": SineTransformer(),
    "SQRT": SqrtTransformer(),
    "SQUARE": SquareTransformer(),
    "RECIPROCAL": ReciprocalTransformer(),
    # "TAN": TanTransformer(),
}
