import doctest
from functools import partial
import pandas as pd
import unittest
from standardize_df import standards
from standardize_df.standards import Standards
import numpy as np
from numpy.testing import assert_equal


class Counter:
    """
    Decorator that counts the number of times the callable was called.
    """
    def __init__(self, fn):
        self.fn = fn
        self._count = 0

    @property
    def count(self):
        return self._count

    def __call__(self, *args, **kwargs):
        self._count += 1
        return self.fn(*args, **kwargs)

    def reset_count(self):
        self._count = 0


def upper_letters(letters: pd.Series):
    return letters.str.upper()


def increment_numbers(numbers: pd.Series):
    return numbers + 1  # adds +1 to every value in the series.


def increment_chars_and_numbers(df: pd.DataFrame):
    # increment letters by 1
    df['letters'] = df['letters'].apply(lambda x: chr(ord(x) + 1))

    # increment numbers by 1
    df['numbers'] = df['numbers'] + 1

    return df


class TestStandardsClass(unittest.TestCase):
    def test_empty_standards_is_ok(self):
        Standards()

    def test_non_callable_values_are_caught_by__init__(self):
        standards_mapping = {
            'letters': 'IM TOTALLY A CALLABLE',
            'numbers': increment_numbers
        }
        self.assertRaises(TypeError, Standards, d_=standards_mapping)

    def test_non_callable_values_are_caught_by_implicit_uses_of___setitem__(self):
        def assign_non_callable_values(standards):
            standards['letters'] = 'IM TOTALLY A CALLABLE'

        def update_non_callable_values(standards):
            standards.update({'letters': 'IM TOTALLY A CALLABLE'})

        def setdefault_non_callable_values(standards):
            standards.setdefault('letters', 'IM TOTALLY A CALLABLE')

        stds = Standards()
        self.assertRaises(TypeError, assign_non_callable_values, stds)
        self.assertRaises(TypeError, update_non_callable_values, stds)
        self.assertRaises(TypeError, setdefault_non_callable_values, stds)

    def test_multi_fields_keys_duplicates_removal(self):
        standards_mapping = {
            'letters': upper_letters,
            ('letters', 'letters', 'numbers'): increment_chars_and_numbers  # multi-field key
        }

        # Default method of remove_dupe should remove one 'letters' from the
        # multi fields key:
        stds = Standards(standards_mapping)
        self.assertIn(('letters', 'numbers'), stds)

        # No remove_dupe should not remove any duplicates from a multi
        # fields key.
        stds = Standards(standards_mapping, remove_dupe=None)
        self.assertIn(('letters', 'letters', 'numbers'), stds)

        # Custom remove_dupe should be applied to the multi fields key:
        def remove_dupes(iterable):
            return tuple(dict.fromkeys(iterable).keys())

        count_calls = Counter(remove_dupes)
        stds = Standards(standards_mapping, remove_dupe=count_calls)
        self.assertEqual(count_calls.count, 1)
        self.assertIn(('letters', 'numbers'), stds)
        self.assertNotIn(('letters', 'letters', 'numbers'), stds)

        # remove_dupe should be applied to multi fields key through different
        # methods that implicitly uses __setitem__: update() and setdefault().
        count_calls.reset_count()
        stds = Standards(remove_dupe=count_calls)
        stds.update({('letters', 'letters', 'numbers'): increment_chars_and_numbers})
        stds.setdefault(('letters', 'letters', 'numbers'),increment_chars_and_numbers)
        self.assertEqual(count_calls.count, 2)

    def test_multi_fields_key_sorts(self):
        def foo(): pass
        standards_mapping = {
            'letters': foo,
            ('numbers', 'letters'): foo,
            ('a', 'c', 'b'): foo
        }
        # Default sorting results in the correct ordering for multi fields
        # keys only.
        stds = Standards(standards_mapping)
        self.assertIn('letters', stds)
        self.assertIn(('letters', 'numbers'), stds)
        self.assertIn(('a', 'b', 'c'), stds)

        # Sorting result does not change ordering of key-value pair ordering:
        self.assertEqual(
            ['letters', ('letters', 'numbers'), ('a', 'b', 'c')],
            list(stds)
        )

        # Custom sorting is applied to multi fields keys.
        reversed = partial(sorted, reverse=True)
        stds = Standards(standards_mapping, sort=reversed)
        self.assertIn('letters', stds)
        self.assertIn(('numbers', 'letters'), stds)
        self.assertIn(('c', 'b', 'a'), stds)

        # Sorting can be turned off with sort=None
        stds = Standards(standards_mapping, sort=None)
        self.assertIn('letters', stds)
        self.assertIn(('numbers', 'letters'), stds)
        self.assertIn(('a', 'c', 'b'), stds)

    def test_multi_fields_keys_as_single(self):
        def foo(): pass
        standards_mapping = {
            ('single_item',): foo,
            ('numbers', 'numbers'): foo,
        }
        # Default processing, after removal of duplicates, keys is a one item
        # iterable. So use that one item as the key:
        stds = Standards(standards_mapping)
        self.assertIn('single_item', stds)
        self.assertIn('numbers', stds)

        # We can turn it off and it should preserve the single item in its
        # iterable.
        stds = Standards(standards_mapping, as_single=False)
        self.assertIn(('single_item',), stds)
        self.assertIn(('numbers',), stds)

    def test_dunder_methods(self):
        def foo(): pass
        standards_mapping = {
            'letters': foo,
            ('letters', 'numbers'): foo
        }
        stds = Standards(standards_mapping)

        # test __len__
        self.assertEqual(len(stds), 2)

        # test __getitem__
        self.assertEqual(stds['letters'], foo)

        # test __setitem__

        # test __delitem__
        del stds['letters']
        self.assertRaises(KeyError, lambda: stds['letters'])

        # test __iter__
        from collections.abc import Iterator
        self.assertIsInstance(iter(stds), Iterator)

        # test __str__

        # test __repr__

    def test_standardize_df(self):
        standards_mapping = {
            'letters': upper_letters,
            ('letters', 'numbers'): increment_chars_and_numbers
        }
        stds = Standards(standards_mapping)
        data = {
            'letters': ['a', 'b', 'c'],
            'numbers': [1, 2, 3]
        }
        df = pd.DataFrame(data)
        standardized_df = stds.standardize_df(df)

        # Should upercase letters and increment letters by 1.
        self.assertEqual(
            standardized_df['letters'].to_list(),
            ['B', 'C', 'D']
        )

        # Should have incremented numbers by 1.
        self.assertEqual(
            standardized_df['numbers'].to_list(),
            [2, 3, 4]
        )

    def test_standardize_df_row_filtering(self):
        def filter_rows_single_field(col: pd.Series):
            return col.drop(index=0)

        def filter_rows_multi_fields(df: pd.DataFrame):
            return df.drop(index=0)

        df = pd.DataFrame(
            data={'letters': ['a', 'b', 'c'], 'numbers': [1, 2, 3]},
            index=[0, 1, 2]
        )

        desired_frame = pd.DataFrame(
            data={'letters': ['b', 'c'], 'numbers': [2, 3]},
            index=[1, 2]
        )
        # Filtering out first row from the single field standards makes
        # appropriate changes to the resulting dataframe.
        standards_mapping = {
            'letters': filter_rows_single_field,
        }
        stds = Standards(standards_mapping)
        standardized_df = stds.standardize_df(df)
        self.assertEqual(standardized_df.to_dict(), desired_frame.to_dict())

        # Filtering out first row from the multi fields standards makes
        # appropriate changes to the resulting dataframe.
        standards_mapping = {
            ('numbers', 'letters'): filter_rows_multi_fields,
        }
        stds = Standards(standards_mapping)
        standardized_df = stds.standardize_df(df)
        self.assertEqual(standardized_df.to_dict(), desired_frame.to_dict())

    def test_standardize_df_row_adding(self):

        def add_rows_single_field(col: pd.Series):
            col[3] = 'd'
            return col

        def add_rows_multi_fields(df: pd.DataFrame):
            df.loc[3] = pd.Series([np.nan, 'd'], index=['numbers', 'letters'])
            return df

        df = pd.DataFrame(
            data={'letters': ['a', 'b', 'c'], 'numbers': [1, 2, 3]},
            index=[0, 1, 2]
        )

        desired_frame = pd.DataFrame(
            data={'letters': ['a', 'b', 'c', 'd'], 'numbers': [1, 2, 3, np.nan]},
            index=[0, 1, 2, 3]
        )

        # Adding a new row from the single field standards makes appropriate
        # changes to the resulting dataframe.
        standards_mapping = {
            'letters': add_rows_single_field,
        }
        stds = Standards(standards_mapping)
        standardized_df = stds.standardize_df(df)
        assert_equal(standardized_df.to_dict(), desired_frame.to_dict())

        # Filtering out first row from the multi fields standards makes
        # changes to the appropriate changes to the resulting dataframe.
        standards_mapping = {
            ('numbers', 'letters'): add_rows_multi_fields,
        }
        stds = Standards(standards_mapping)
        standardized_df = stds.standardize_df(df)
        assert_equal(standardized_df.to_dict(), desired_frame.to_dict())

    def test_standardize_df_rows_overriding(self):
        def override_letters(col: pd.Series):
            col[0] = 'z'
            return col

        df = pd.DataFrame(
            data={'letters': ['z', 'b', 'c'], 'numbers': [1, 2, 3]},
        )

        desired_frame = pd.DataFrame(
            data={'letters': ['z', 'b', 'c'], 'numbers': [1, 2, 3]},
        )

        stds = Standards()
        stds['letters'] = override_letters
        standardized_df = stds.standardize_df(df)

        assert_equal(
            standardized_df.to_dict(),
            desired_frame.to_dict()
        )

    def test_standardize_df_columns_filtering(self):
        def drop_columns_single_field(col: pd.Series):
            """Drops col in favor of the 'signs' column"""
            return pd.Series(['!', '@', '#'], name='signs')

        def drop_columns_multi_fields(df: pd.DataFrame):
            """Drops the letters column."""
            del df['letters']
            return df

        df = pd.DataFrame(
            data={'letters': ['a', 'b', 'c'], 'numbers': [1, 2, 3]},
        )

        desired_frame_single = pd.DataFrame(
            data={'numbers': [1, 2, 3], 'signs': ['!', '@', '#']}
        )
        desired_frame_multi = pd.DataFrame(data={'numbers': [1, 2, 3]})

        # Dropping or replacing a column from the single field standards makes
        # the appropriate changes to the resulting dataframe.
        standards_mapping = {
            'letters': drop_columns_single_field,
        }
        stds = Standards(standards_mapping)
        standardized_df = stds.standardize_df(df)
        assert_equal(standardized_df.to_dict(), desired_frame_single.to_dict())

        # Dropping or replacing a column from the multi fields standards makes
        # the appropriate changes to the resulting dataframe.
        standards_mapping = {
            ('numbers', 'letters'): drop_columns_multi_fields,
        }
        stds = Standards(standards_mapping)
        standardized_df = stds.standardize_df(df)
        assert_equal(standardized_df.to_dict(), desired_frame_multi.to_dict())

    def test_standardize_df_columns_overriding(self):
        def override_letters(col: pd.Series):
            return pd.Series(['x', 'y', 'z'], name='letters')

        df = pd.DataFrame(
            data={'letters': ['a', 'b', 'c'], 'numbers': [1, 2, 3]},
        )

        desired_frame = pd.DataFrame(
            data={'letters': ['x', 'y', 'z'], 'numbers': [1, 2, 3]},
        )

        stds = Standards()
        stds['letters'] = override_letters
        standardized_df = stds.standardize_df(df)

        assert_equal(
            standardized_df.to_dict(),
            desired_frame.to_dict()
        )

    def test_standardize_df_columns_adding(self):
        def add_columns_single_field(col: pd.Series):
            df = pd.DataFrame(
                {'signs': ['!', '@', '#']}
            )
            df['letters'] = col
            return df

        def add_columns_multi_fields(df: pd.DataFrame):
            df['signs'] = ['!', '@', '#']
            return df

        df = pd.DataFrame(
            data={'letters': ['a', 'b', 'c'], 'numbers': [1, 2, 3]},
        )

        desired_frame = pd.DataFrame(
            data={
                'letters': ['a', 'b', 'c'],
                'numbers': [1, 2, 3],
                'signs': ['!', '@', '#']
            },
        )

        # Adding a new column from the single field standards makes appropriate
        # changes to the resulting dataframe.
        standards_mapping = {
            'letters': add_columns_single_field,
        }
        stds = Standards(standards_mapping)
        standardized_df = stds.standardize_df(df)
        assert_equal(standardized_df.to_dict(), desired_frame.to_dict())

        # Adding a new column from the multi field standards makes appropriate
        # changes to the resulting dataframe.
        standards_mapping = {
            ('numbers', 'letters'): add_columns_multi_fields,
        }
        stds = Standards(standards_mapping)
        standardized_df = stds.standardize_df(df)
        assert_equal(standardized_df.to_dict(), desired_frame.to_dict())

    def test_empty_result_returns_empty_dataframes_drops_columns_and_not_indexes(self):
        def return_empty_series(cols):
            return pd.Series()

        def return_empty_df(df):
            return pd.DataFrame()

        df = pd.DataFrame(
            data={'letters': ['a', 'b', 'c'], 'numbers': [1, 2, 3]},
        )

        desired_frame_single_field = pd.DataFrame(data={'numbers': [1, 2, 3]})

        # standardized subsets that return as an empty series or dataframe
        # only drops the columns associated with the original (pre-standardized)
        # subset.

        stds = Standards()
        stds['letters'] = return_empty_series
        standardized_df = stds.standardize_df(df)
        assert_equal(
            standardized_df.to_dict(),
            desired_frame_single_field.to_dict()
        )
        stds['letters'] = return_empty_df
        standardized_df = stds.standardize_df(df)
        assert_equal(
            standardized_df.to_dict(),
            desired_frame_single_field.to_dict()
        )

        desired_frame_multi_fields = pd.DataFrame()

        # Multi-fields standards returning empty series or dataframes drops
        # all fields in the multi-fields.

        stds = Standards()
        stds[('letters', 'numbers')] = return_empty_series
        standardized_df = stds.standardize_df(df)
        assert_equal(
            standardized_df.to_dict(),
            desired_frame_multi_fields.to_dict()
        )

        stds[('letters', 'numbers')] = return_empty_series
        standardized_df = stds.standardize_df(df)
        assert_equal(
            standardized_df.to_dict(),
            desired_frame_multi_fields.to_dict()
        )

    def test_standardize_shared_only(self):
        def return_arg(x):
            return x

        df = pd.DataFrame(
            data={'letters': ['a', 'b', 'c'], 'numbers': [1, 2, 3]},
        )
        desired_frame = df.copy()

        standards_mapping = {
            'letters': return_arg,
            ('letters', 'numbers'): return_arg,
            'non_existing_field1': return_arg,
            ('non_existing_field2', 'non_existing_field3'): return_arg,
        }
        stds = Standards(standards_mapping)

        # With shared_only=False, standardize_df() raises a KeyError if there
        # are fields from a standards instance missing field from the
        # dataframe.
        self.assertRaises(
            KeyError, stds.standardize_df, df=df, shared_only=False
        )
        try:
            stds.standardize_df(df, shared_only=False)
        except KeyError as err:
            self.assertIn('non_existing_field', str(err))

        # With shared_only=True, standardize_df() simply ignores running
        # standardization callables paired to field(s) missing from the
        # dataframe.
        standardized_df = stds.standardize_df(df, shared_only=True)
        assert_equal(standardized_df.to_dict(), desired_frame.to_dict())

    def test_shared_only_is_ignored_if_column_is_created_during_standardization(self):
        # If restrict_cols=False, then callables associated with a
        # missing field from the dataframe will be called if previous
        # operations have created that column, regardless if shared_only
        # argument is true or false.

        def add_symbols(series):
            '''adds a symbols column to the dataframe.'''
            data = {'letters': ['a', 'b', 'c'], 'symbols':['!', '@', '#']}
            return pd.DataFrame(data)

        def increment_symbols(series):
            '''increments symbols columns by 1 character.'''
            return series.apply(lambda x: chr(ord(x) + 1))


        df = pd.DataFrame(
            data={'letters': ['a', 'b', 'c'], 'numbers': [1, 2, 3]},
        )
        desired_frame = pd.DataFrame(
            {
                'letters': ['a', 'b', 'c'],
                'numbers':[1, 2, 3],
                'symbols':['\"', 'A', '$']
            }
        )

        standards_mapping = {
            'letters': add_symbols,
            'symbols': increment_symbols,
        }

        stds = Standards(standards_mapping)
        standardized_df = stds.standardize_df(df, shared_only=True)
        assert_equal(standardized_df.to_dict(), desired_frame.to_dict())
        standardized_df = stds.standardize_df(df, shared_only=False)
        assert_equal(standardized_df.to_dict(), desired_frame.to_dict())

    def test_restrict_cols(self):
        def add_symbols(series):
            data = {'letters': ['a', 'b', 'c'], 'symbols': ['!', '@', '#']}
            return pd.DataFrame(data)

        def drop_letters(series):
            return pd.Series()

        data = {'letters': ['a', 'b', 'c'], 'numbers': [1, 2, 3]}
        df = pd.DataFrame(data)
        stds = Standards()

        stds['letters'] = add_symbols
        self.assertRaises(
            ValueError,
            stds.standardize_df,
            df,
            restrict_cols=True
        )

        stds['letters'] = drop_letters
        self.assertRaises(
            ValueError,
            stds.standardize_df,
            df,
            restrict_cols=True
        )

    def test_restrict_rows(self):
        def add_d(series):
            return pd.Series(['a', 'b', 'c', 'd'], name='letters')

        def drop_a(series):
            return series[lambda x: x != 'a']

        def change_index(series):
            series.index = [2, 3, 4]
            return series

        data = {'letters': ['a', 'b', 'c'], 'numbers': [1, 2, 3]}
        df = pd.DataFrame(data)
        stds = Standards()

        # restrict_rows=True restricts adding of rows
        stds['letters'] = add_d
        self.assertRaises(
            ValueError,
            stds.standardize_df,
            df,
            restrict_rows=True
        )

        # restrict_rows=True restricts rows dropping
        stds['letters'] = drop_a
        self.assertRaises(
            ValueError,
            stds.standardize_df,
            df,
            restrict_rows=True
        )

        # restrict_rows=True restricts rows changing (indexes)
        stds['letters'] = change_index
        self.assertRaises(
            ValueError,
            stds.standardize_df,
            df,
            restrict_rows=True
        )


def load_tests(loader, tests, ignore):
    """Import doctests"""
    tests.addTests(doctest.DocTestSuite(standards))
    return tests


if __name__ == '__main__':
    unittest.main()
