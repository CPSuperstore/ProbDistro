import typing
import ProbDistro.base_discrete_distribution as base_discrete_distribution


class DiscreteRandomVariable(base_discrete_distribution.BaseDiscreteDistribution):
    def __init__(self, x: typing.Sequence[float], px: typing.Sequence[float]):
        if sum(px) != 1:
            raise ValueError(
                "Sum of all probabilities must equal 1 by law of total probability. Got {}.".format(sum(px))
            )

        if len(x) != len(px):
            raise ValueError("There are {} x values, but {} probabilities. These counts must match.".format(
                len(x), len(px)
            ))

        self.x = list(x)
        self.px = list(px)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(list(data.keys()), list(data.values()))

    def __pow__(self, power, modulo=None) -> 'DiscreteRandomVariable':
        return DiscreteRandomVariable([x ** power for x in self.x], self.px)

    def pmf(self, x: float) -> float:
        return self.px[self.x.index(x)]

    def cdf(self, x: float) -> float:
        total = 0
        for i in range(len(self.x)):
            if self.x[i] <= x:
                total += self.px[i]

        return total

    def _is_supported(self, x: float) -> bool:
        return x in self.x

    def expected_value(self) -> float:
        return sum(a * b for a, b in zip(self.x, self.px))

    def variance(self) -> float:
        return (self ** 2).expected_value() - self.expected_value() ** 2
