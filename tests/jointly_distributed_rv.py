import sys
import unittest

sys.path.append('..')

from ProbDistro import *


class TestJointlyDistributedRV(unittest.TestCase):
    def test_and(self):
        a = DiscreteRandomVariable([1, 2, 3, 4], [0.05, 0.125, 0.525, 0.3])
        b = DiscreteRandomVariable([2, 4, 6], [0.3, 0.6, 0.1])

        self.assertEqual(a.p_and(a(1), b(4)), 0.03)

    def test_or(self):
        a = DiscreteRandomVariable([1, 2, 3, 4], [0.05, 0.125, 0.525, 0.3])
        b = DiscreteRandomVariable([2, 4, 6], [0.3, 0.6, 0.1])

        self.assertEqual(a.p_or(a(1), b(4)), 0.62)

    def test_given(self):
        a = DiscreteRandomVariable([1, 2, 3, 4], [0.05, 0.125, 0.525, 0.3])
        b = DiscreteRandomVariable([2, 4, 6], [0.3, 0.6, 0.1])

        self.assertEqual(a.p_given(a(1), b(4)), 0.6)

    def test_joint_distribution(self):
        a = DiscreteRandomVariable([1, 2, 3, 4], [0.05, 0.125, 0.525, 0.3])
        b = DiscreteRandomVariable([2, 4, 6], [0.3, 0.6, 0.1])

        self.assertEqual(
            a.jointly_distributed_table(b),
            [
                [0.015, 0.0375, 0.1575, 0.09],
                [0.03, 0.075, 0.315, 0.18],
                [0.005000000000000001, 0.0125, 0.052500000000000005, 0.03]
            ]
        )

    def test_multiplication(self):
        a = DiscreteRandomVariable([1, 2, 3, 4], [0.05, 0.125, 0.525, 0.3])
        b = DiscreteRandomVariable([2, 4, 6], [0.3, 0.6, 0.1])

        ab = a * b

        self.assertEqual(ab.x, [2, 4, 6, 8, 12, 16, 18, 24])
        self.assertEqual(ab.px, [0.015, 0.0675, 0.1625, 0.16499999999999998, 0.3275, 0.18, 0.052500000000000005, 0.03])

    def test_covariance(self):
        a = DiscreteRandomVariable([1, 2, 3, 4], [0.05, 0.125, 0.525, 0.3])
        b = DiscreteRandomVariable([2, 4, 6], [0.3, 0.6, 0.1])

        self.assertEqual(a.covariance(b), 1.7763568394002505e-15)

    def test_correlation(self):
        a = DiscreteRandomVariable([1, 2, 3, 4], [0.05, 0.125, 0.525, 0.3])
        b = DiscreteRandomVariable([2, 4, 6], [0.3, 0.6, 0.1])

        self.assertEqual(a.correlation(b), 1.8809278223539317e-15)


if __name__ == '__main__':
    unittest.main()
