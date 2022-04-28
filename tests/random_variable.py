import sys
import unittest

sys.path.append('..')

from ProbDistro import *


class TestRandomVariable(unittest.TestCase):
    def test_constructor(self):
        self.assertRaises(
            ValueError,
            lambda: RandomVariable([1, 2, 3], [0.5, 0.25, 0.5]),
        )

        self.assertRaises(
            ValueError,
            lambda: RandomVariable([1, 2, 3], [0.5, 0.25]),
        )

        self.assertRaises(
            ValueError,
            lambda: RandomVariable([1, 2], [0.5, 0.25, 0.25]),
        )

        RandomVariable([1, 2, 3], [0.5, 0.25, 0.25])

    def test_expected_value(self):
        p = RandomVariable([1, 2, 3], [0.5, 0.25, 0.25])

        self.assertEqual(p.expected_value(), 1.75)
        self.assertEqual(p.mean(), 1.75)

    def test_exponent(self):
        p = RandomVariable([1, 2, 3], [0.5, 0.25, 0.25]) ** 2
        self.assertEqual(p.x, [1, 4, 9])

        p = RandomVariable([1, 2, 3], [0.5, 0.25, 0.25]) ** 3
        self.assertEqual(p.x, [1, 8, 27])

    def test_variance(self):
        p = RandomVariable([1, 2], [0.4, 0.6])
        self.assertEqual(round(p.variance(), 4), 0.24)

        p = RandomVariable([1, 2], [0.5, 0.5])
        self.assertEqual(round(p.variance(), 4), 0.25)


if __name__ == '__main__':
    unittest.main()
