import math
import sys
import unittest

sys.path.append('..')

from ProbDistro import *


class TestGeometricWithSuccess(unittest.TestCase):
    def test_pmf(self):
        p = Geometric(0.75, include_success_trial=True)

        self.assertEqual(p.pmf(1), 0.75)
        self.assertEqual(p.pmf(3), 0.046875)
        self.assertEqual(p.pmf(8), 4.57763671875e-05)

        self.assertRaises(ValueError, lambda: p(0))
        self.assertRaises(ValueError, lambda: p(0.5))
        self.assertRaises(ValueError, lambda: p(-1))

    def test_cdf(self):
        p = Geometric(0.75, include_success_trial=True)

        self.assertEqual(p.cdf(1), 0.75)
        self.assertEqual(p.cdf(3), 0.984375)
        self.assertEqual(p.cdf(8), 0.9999847412109375)

    def test_properties(self):
        p = Geometric(0.75, include_success_trial=True)

        self.assertEqual(p.expected_value(), p.mean())
        self.assertEqual(p.expected_value(), 4 / 3)
        self.assertEqual(p.variance(), 4 / 9)
        self.assertEqual(round(p.standard_deviation(), 5), round(math.sqrt(4 / 9), 5))

    def test_to_discrete_rv(self):
        p = Geometric(0.75, include_success_trial=True)

        self.assertRaises(ValueError, lambda: p.to_discrete_random_variable())

        rv = p.to_discrete_random_variable(stop=5)

        self.assertEqual(rv.x, [1, 2, 3, 4, 5])
        self.assertEqual(
            rv.px,
            [0.750733137829912, 0.187683284457478, 0.0469208211143695, 0.011730205278592375, 0.002932551319648094]
        )


if __name__ == '__main__':
    unittest.main()
