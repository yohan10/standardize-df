import doctest
import unittest
from standardize_df import utils


def load_tests(loader, tests, ignore):
    """Import doctests"""
    tests.addTests(doctest.DocTestSuite(utils))
    return tests


if __name__ == '__main__':
    unittest.main()