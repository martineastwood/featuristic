FITNESS_REGISTRY = {}


def register_fitness(name):
    def decorator(func):
        FITNESS_REGISTRY[name] = func
        return func

    return decorator


def get_fitness(name):
    return FITNESS_REGISTRY[name]


def list_fitness_functions():
    return list(FITNESS_REGISTRY.keys())
