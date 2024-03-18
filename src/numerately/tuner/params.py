import numpy as np


class Param:
    def __init__(self):
        pass

    def sample(self):
        raise NotImplementedError


class Fixed(Param):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value


class Uniform(Param):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self):
        return np.random.uniform(self.low, self.high)


class LogUniform(Param):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self):
        return np.exp(np.random.uniform(np.log(self.low), np.log(self.high)))


class Categorical(Param):
    def __init__(self, choices):
        self.choices = choices

    def sample(self):
        return np.random.choice(self.choices)


class Int(Param):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self):
        return np.random.randint(self.low, self.high + 1)


class Bool(Param):
    def sample(self):
        return np.random.choice([True, False])
