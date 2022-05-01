import sys
import unittest

sys.path.append('..')

from ProbDistro import *


class TestRandomVariable(unittest.TestCase):
    def test_constructor(self):
        self.assertRaises(
            ValueError, lambda: DiscreteRandomVariable(
                [1, 2, 3, 4],
                [0.5, 0.25, 0.125, 0]
            )
        )

        self.assertRaises(
            ValueError, lambda: DiscreteRandomVariable(
                [1, 2, 3],
                [0.5, 0.25, 0.125, 0.125]
            )
        )

        self.assertRaises(
            ValueError, lambda: DiscreteRandomVariable(
                [1, 2, 3, 4],
                [0.5, 0.25, 0.25]
            )
        )

        DiscreteRandomVariable(
            [1, 2, 3, 4],
            [0.5, 0.25, 0.125, 0.125]
        )

    def test_pmf(self):
        x = [1, 2, 3, 4]
        px = [0.5, 0.25, 0.125, 0.125]
        a = DiscreteRandomVariable(x, px)

        self.assertRaises(ValueError, lambda: a.pmf(5))
        self.assertRaises(ValueError, lambda: a.pmf(0))

        for i, pi in zip(x, px):
            self.assertEqual(a.pmf(i), pi)

    def test_cdf(self):
        x = [1, 2, 3, 4]
        px = [0.5, 0.25, 0.125, 0.125]
        a = DiscreteRandomVariable(x, px)

        self.assertEqual(a.cdf(0), 0)

        total = 0
        for i, pi in zip(x, px):
            total += pi
            self.assertEqual(a.cdf(i), total)

        self.assertEqual(a.cdf(5), 1)

    def test_expected_value(self):
        x = [1, 2, 3, 4]
        px = [0.5, 0.25, 0.125, 0.125]
        a = DiscreteRandomVariable(x, px)

        self.assertEqual(a.expected_value(), 1.875)
        self.assertEqual(a.mean(), 1.875)

    def test_variance(self):
        p = DiscreteRandomVariable([1, 2], [0.4, 0.6])
        self.assertEqual(round(p.variance(), 4), 0.24)

        p = DiscreteRandomVariable([1, 2], [0.5, 0.5])
        self.assertEqual(round(p.variance(), 4), 0.25)

    def test_exponent(self):
        p = DiscreteRandomVariable([1, 2, 3], [0.5, 0.25, 0.25]) ** 2
        self.assertEqual(p.x, [1, 4, 9])

        p = DiscreteRandomVariable([1, 2, 3], [0.5, 0.25, 0.25]) ** 3
        self.assertEqual(p.x, [1, 8, 27])


if __name__ == '__main__':
    unittest.main()
