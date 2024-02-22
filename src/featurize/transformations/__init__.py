from . import numeric
from . import combinations
import pandas as pd


def list_transformations() -> pd.DataFrame:
    output = []

    for _, v in numeric.transfomers.items():
        tmp = {
            "name": v.name,
            "type": "numeric",
            "description": v.get_description(),
        }
        output.append(tmp)

    for _, v in combinations.transformers.items():
        tmp = {
            "name": v.name,
            "type": "combinations",
            "description": v.get_description(),
        }
        output.append(tmp)

    output = pd.DataFrame(output)
    return output
