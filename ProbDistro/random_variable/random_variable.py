import math


class RandomVariable:
    def __init__(self, x: list, px: list):
        self.x = x
        self.px = px

        if sum(self.px) != 1:
            raise ValueError("Sum of all probabilities must equal 1")

        if len(self.x) != len(self.px):
            raise ValueError("Length of x ({}) must match length of px ({})".format(len(self.x), len(self.px)))

    def __repr__(self):
        return "<RandomVariable x={} p(X=x)={}>".format(self.x, self.px)

    def __pow__(self, power, modulo=None) -> 'RandomVariable':
        return RandomVariable([x ** power for x in self.x], self.px)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(list(data.keys()), list(data.values()))

    def mean(self) -> float:
        return self.expected_value()

    def expected_value(self) -> float:
        return sum(a * b for a, b in zip(self.x, self.px))

    def variance(self) -> float:
        return (self ** 2).expected_value() - self.expected_value() ** 2

    def standard_deviation(self) -> float:
        return math.sqrt(self.variance())
