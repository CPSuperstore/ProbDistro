import math
import sys
import unittest

sys.path.append('..')

from ProbDistro import *


class TestBinomial(unittest.TestCase):
    def test_pmf(self):
        p = Binomial(10, 0.5)

        self.assertEqual(p.pmf(5), 0.24609375)
        self.assertEqual(p.pmf(10), 0.0009765625)

        self.assertRaises(ValueError, lambda: p(0.5))
        self.assertRaises(ValueError, lambda: p(-1))
        self.assertRaises(ValueError, lambda: p(11))

    def test_cdf(self):
        p = Binomial(10, 0.5)

        self.assertEqual(p.cdf(-1), 0)
        self.assertEqual(p.cdf(5), 0.623046875)
        self.assertEqual(p.cdf(10), 1)

    def test_properties(self):
        p = Binomial(10, 0.5)

        self.assertEqual(p.expected_value(), p.mean())
        self.assertEqual(p.expected_value(), 5)
        self.assertEqual(p.variance(), 2.5)
        self.assertEqual(round(p.standard_deviation(), 5), round(math.sqrt(2.5), 5))

    def test_to_discrete_rv(self):
        p = Binomial(10, 0.5)
        rv = p.to_discrete_random_variable()
        self.assertEqual(rv.x, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(
            rv.px, [
                0.0009765625, 0.009765625, 0.0439453125, 0.1171875, 0.205078125, 0.24609375, 0.205078125,
                0.1171875, 0.0439453125, 0.009765625, 0.0009765625
            ]
        )


if __name__ == '__main__':
    unittest.main()
