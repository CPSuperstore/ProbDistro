import sys
import unittest

sys.path.append('..')

from ProbDistro import *


class TestHypergeometric(unittest.TestCase):
    def test_pmf(self):
        p = Hypergeometric(10, 5, 3)

        self.assertEqual(p.pmf(0), 0.08333333333333333)
        self.assertEqual(p.pmf(1), 0.4166666666666667)
        self.assertEqual(p.pmf(2), 0.4166666666666667)
        self.assertEqual(p.pmf(3), 0.08333333333333333)

        self.assertRaises(ValueError, lambda: p(0.5))
        self.assertRaises(ValueError, lambda: p(-1))
        self.assertRaises(ValueError, lambda: p(4))

    def test_cdf(self):
        p = Hypergeometric(10, 5, 3)

        self.assertEqual(p.cdf(-1), 0)

        self.assertEqual(p.cdf(0), 0.08333333333333333)
        self.assertEqual(p.cdf(1), 0.5)
        self.assertEqual(p.cdf(2), 0.9166666666666667)
        self.assertEqual(p.cdf(3), 1)

        self.assertEqual(p.cdf(4), 1)

    def test_properties(self):
        p = Hypergeometric(10, 5, 3)

        self.assertEqual(p.expected_value(), p.mean())
        self.assertEqual(p.expected_value(), 1.5)
        self.assertEqual(p.variance(), 0.5833333333333334)
        self.assertEqual(p.standard_deviation(), 0.7637626158259734)

    def test_to_discrete_rv(self):
        p = Hypergeometric(10, 5, 3)
        rv = p.to_discrete_random_variable()
        self.assertEqual(rv.x, [0, 1, 2, 3])
        self.assertEqual(
            rv.px, [0.08333333333333333, 0.4166666666666667, 0.4166666666666667, 0.08333333333333333]
        )


if __name__ == '__main__':
    unittest.main()
