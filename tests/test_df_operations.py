import doctest
import unittest
from standardize_df import df_operations


def load_tests(loader, tests, ignore):
    """Import doctests"""
    tests.addTests(doctest.DocTestSuite(df_operations))
    return tests


if __name__ == '__main__':
    unittest.main()
