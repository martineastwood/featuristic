from .divide import DivideTransformer
from .multiply import MultiplyTransformer
from .minus import SubtractTransformer
from .plus import PlusTransformer

transformers = {
    # "DIVIDE": DivideTransformer(),
    # "MULTIPLY": MultiplyTransformer(),
    "MINUS": SubtractTransformer(),
    "PLUS": PlusTransformer(),
}
