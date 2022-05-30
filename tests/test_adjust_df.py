import doctest
import unittest
from standardize_df import adjust_df


def load_tests(loader, tests, ignore):
    """Import doctests"""
    tests.addTests(doctest.DocTestSuite(adjust_df))
    return tests


if __name__ == '__main__':
    unittest.main()
