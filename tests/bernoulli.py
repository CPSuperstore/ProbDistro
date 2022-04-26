import math
import sys
import unittest

sys.path.append('..')

from ProbDistro import *


class TestBernoulli(unittest.TestCase):
    def test_pmf(self):
        p = Bernoulli(0.7)

        self.assertEqual(round(p.pmf(0), 2), 0.3)
        self.assertEqual(p.pmf(1), 0.7)

        self.assertRaises(ValueError, lambda: p(0.5))
        self.assertRaises(ValueError, lambda: p(-1))
        self.assertRaises(ValueError, lambda: p(2))

    def test_cdf(self):
        p = Bernoulli(0.7)

        self.assertEqual(p.cdf(-1), 0)
        self.assertEqual(round(p.cdf(0.1), 2), 0.3)
        self.assertEqual(round(p.cdf(0.9), 2), 0.3)
        self.assertEqual(p.cdf(2), 1)

    def test_properties(self):
        p = Bernoulli(0.7)

        self.assertEqual(p.expected_value(), p.mean())
        self.assertEqual(p.expected_value(), 0.7)
        self.assertEqual(round(p.variance(), 2), 0.21)
        self.assertEqual(round(p.standard_deviation(), 5), round(math.sqrt(0.21), 5))


if __name__ == '__main__':
    unittest.main()
