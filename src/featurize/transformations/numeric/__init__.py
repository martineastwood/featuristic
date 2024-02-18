from .absolute import AbsoluteTransformer
from .cosine import CosineTransformer
from .cube import CubeTransformer
from .percentile import PercentileTransformer
from .sine import SineTransformer
from .sqrt import SqrtTransformer
from .square import SquareTransformer
from .tan import TanTransformer
from .reciprocal import ReciprocalTransformer


transfomers = {
    "ABSOLUTE": AbsoluteTransformer(),
    "COSINE": CosineTransformer(),
    "CUBE": CubeTransformer(),
    "PERCENTILE": PercentileTransformer(),
    "SINE": SineTransformer(),
    "SQRT": SqrtTransformer(),
    "SQUARE": SquareTransformer(),
    "TAN": TanTransformer(),
    "RECIPROCAL": ReciprocalTransformer(),
}
