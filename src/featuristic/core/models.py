from dataclasses import dataclass


@dataclass(order=True)
class ProgramFitness:
    fitness: float
    individual: dict
    name: str = ""
