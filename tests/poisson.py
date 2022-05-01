import math
import sys
import unittest

sys.path.append('..')

from ProbDistro import *


class TestPoisson(unittest.TestCase):
    def test_pmf(self):
        p = Poisson(0.75)

        self.assertEqual(p.pmf(0), 0.47236655274101474)
        self.assertEqual(p.pmf(2), 0.13285309295841038)
        self.assertEqual(p.pmf(5), 0.0009341233098638231)

        self.assertRaises(ValueError, lambda: p(0.5))
        self.assertRaises(ValueError, lambda: p(-1))

    def test_cdf(self):
        p = Poisson(0.75)

        self.assertEqual(p.cdf(-1), 0)
        self.assertEqual(p.cdf(0), 0.47236655274101474)
        self.assertEqual(p.cdf(2), 0.9594945602551862)
        self.assertEqual(p.cdf(5), 0.9998694455370781)
        self.assertEqual(p.cdf(10), 0.9999999994670575)

    def test_properties(self):
        p = Poisson(0.75)

        self.assertEqual(p.expected_value(), p.mean())
        self.assertEqual(p.expected_value(), 0.75)
        self.assertEqual(p.variance(), 0.75)
        self.assertEqual(round(p.standard_deviation(), 5), round(math.sqrt(0.75), 5))

    def test_to_discrete_rv(self):
        p = Poisson(0.75)

        self.assertRaises(ValueError, lambda: p.to_discrete_random_variable())

        rv = p.to_discrete_random_variable(stop=4)
        self.assertEqual(rv.x, [0, 1, 2, 3, 4])
        self.assertEqual(
            rv.px, [
                0.47287000692680675, 0.35465250519510505, 0.1329946894481644, 0.0332486723620411, 0.006234126067882706
            ]
        )


if __name__ == '__main__':
    unittest.main()
