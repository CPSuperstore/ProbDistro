import math
import sys
import unittest

sys.path.append('..')

from ProbDistro import *


class TestGeometricWithSuccess(unittest.TestCase):
    def test_pmf(self):
        p = Geometric(0.75, include_success_trial=False)

        self.assertEqual(p.pmf(0), 0.75)
        self.assertEqual(p.pmf(1), 0.1875)
        self.assertEqual(p.pmf(3), 0.01171875)
        self.assertEqual(p.pmf(8), 1.1444091796875e-05)

        self.assertRaises(ValueError, lambda: p(0.5))
        self.assertRaises(ValueError, lambda: p(-1))

    def test_cdf(self):
        p = Geometric(0.75, include_success_trial=False)

        self.assertEqual(p.cdf(0), 0.75)
        self.assertEqual(p.cdf(1), 0.9375)
        self.assertEqual(p.cdf(3), 0.99609375)
        self.assertEqual(p.cdf(8), 0.9999961853027344)

    def test_properties(self):
        p = Geometric(0.75, include_success_trial=False)

        self.assertEqual(p.expected_value(), p.mean())
        self.assertEqual(p.expected_value(), 1 / 3)
        self.assertEqual(p.variance(), 4 / 9)
        self.assertEqual(round(p.standard_deviation(), 5), round(math.sqrt(4 / 9), 5))

    def test_to_discrete_rv(self):
        p = Geometric(0.75, include_success_trial=False)

        self.assertRaises(ValueError, lambda: p.to_discrete_random_variable())

        rv = p.to_discrete_random_variable(stop=4)

        self.assertEqual(rv.x, [0, 1, 2, 3, 4])
        self.assertEqual(
            rv.px,
            [0.750733137829912, 0.187683284457478, 0.0469208211143695, 0.011730205278592375, 0.002932551319648094]
        )


if __name__ == '__main__':
    unittest.main()
