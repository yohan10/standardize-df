from functools import wraps
from typing import Iterable, Union
import pandas as pd


def enclose_string_fields(fn):
    """ Encloses string fields with a list."""
    @wraps(fn)
    def inner(df, cols, fields):
        if isinstance(fields, str):
            fields = [fields]
        return fn(df, cols, fields)
    return inner


def add_rows(
        df: pd.DataFrame,
        cols: Union[pd.Series, pd.DataFrame]
) -> pd.DataFrame:
    """
    Adds rows from cols, an altered subset of df, that are missing in df.

    Examples
    --------
    >>> df = pd.DataFrame({'letters': ['a', 'b', 'c']})
    >>> letters = pd.Series(
    ...     data=['w', 'x', 'y', 'z'],
    ...     index=[0, 1, 2, 3],
    ...     name='letters'
    ... )
    >>> add_rows(df, letters)
      letters
    0       a
    1       b
    2       c
    3       z
    >>> letters = pd.DataFrame(
    ...     data={'letters': ['w', 'x', 'y', 'z']},
    ...     index=[4, 5, 6, 7]
    ... )
    >>> add_rows(df, letters)
      letters
    0       a
    1       b
    2       c
    3       w
    4       x
    5       y
    6       z
    """
    added_rows = [idx for idx in cols.index if idx not in df.index]

    if added_rows:
        try:
            name = cols.name
            data = [cols[idx] for idx in added_rows]
            missing_col = pd.Series(
                data,
                name=name,
                index=added_rows
            )
        except AttributeError:
            missing_col = cols[cols.index.isin(added_rows)]
        df = df.merge(missing_col, how="outer")
    return df


def drop_rows(
        df: pd.DataFrame,
        cols: Union[pd.Series, pd.DataFrame]
) -> pd.DataFrame:
    """
    Drops rows from df missing in cols, an altered subset of df.

    Examples
    --------
    >>> df = pd.DataFrame({'letters': ['a', 'b', 'c']})
    >>> letters = pd.Series(
    ...     data=['a', 'b'],
    ...     index=[0, 1],
    ...     name='letters'
    ... )
    >>> drop_rows(df, letters)
      letters
    0       a
    1       b
    >>> letters = pd.DataFrame(
    ...     data={'letters': [2, 3, 4]},
    ...     index=[2, 3, 4]
    ... )
    >>> drop_rows(df, letters)
      letters
    2       c
    """
    dropped_rows = [idx for idx in df.index if idx not in cols.index]
    if dropped_rows:
        df = df[df.index.isin(cols.index)]
    return df


def adjust_rows(
        df: pd.DataFrame,
        cols: Union[pd.Series, pd.DataFrame]
) -> pd.DataFrame:
    """
    Adds rows from cols to df and drops rows in df missing from cols.

    Examples
    --------
    >>> df = pd.DataFrame({'letters': ['a', 'b', 'c']})
    >>> letters = pd.Series(
    ...     data=['a', 'b', 'c', 'd'],
    ...     index=[1, 2, 3, 4],
    ...     name='letters'
    ... )
    >>> adjust_rows(df, letters)
      letters
    1       b
    2       c
    3       d
    >>> letters = pd.DataFrame(
    ...     data={'letters': ['w', 'x', 'y', 'z']},
    ...     index=[1, 2, 3, 4]
    ... )
    >>> adjust_rows(df, letters)
      letters
    1       b
    2       c
    3       y
    4       z
    """
    if cols.empty:
        return df.drop(index=df.index)
    df = add_rows(df, cols)
    df = drop_rows(df, cols)
    return df


def add_columns(
        df: pd.DataFrame,
        cols: Union[pd.Series, pd.DataFrame],
        fields: Union[str, Iterable]
) -> pd.DataFrame:
    """
    Adds columns from cols, an altered subset of df, that are missing in df.

    Examples
    --------
    >>> df = pd.DataFrame({'letters': ['a', 'b', 'c']})
    >>> numbers = pd.Series(
    ...     data=[0, 1, 2],
    ...     name='numbers'
    ... )
    >>> add_columns(df.copy(), numbers, fields=['letters'])
      letters  numbers
    0       a        0
    1       b        1
    2       c        2
    >>> letters_and_nums = pd.DataFrame(
    ...     data={'letters': ['w', 'x', 'y', 'z'], 'numbers': [1, 2, 3, 4]},
    ...     index=[1, 2, 3, 4]
    ... )
    >>> add_columns(df.copy(), letters_and_nums, fields=['letters'])
      letters  numbers
    0       a      NaN
    1       b      1.0
    2       c      2.0
    >>> add_columns(df.copy(), letters_and_nums, fields=[])
      letters
    0       a
    1       b
    2       c
    """
    if fields:
        try:
            missing_left = cols.name if cols.name not in fields else None
            if missing_left:
                df[missing_left] = cols

        except AttributeError:
            missing_left = set(cols.columns) - set(fields)
            if missing_left:
                df[list(missing_left)] = cols[list(missing_left)]
    return df


@enclose_string_fields
def drop_columns(
        df: pd.DataFrame,
        cols: Union[pd.Series, pd.DataFrame],
        fields: Union[str, Iterable]
) -> pd.DataFrame:
    """
    Drops columns from df with any field name in fields missing in cols,
    an altered subset of df.

    Examples
    --------
    >>> df = pd.DataFrame({'letters': ['a', 'b', 'c'], 'numbers': [0, 1, 2]})
    >>> numbers = pd.Series([], name='numbers')
    >>> drop_columns(df.copy(), numbers, fields=['letters', 'numbers'])
       numbers
    0        0
    1        1
    2        2
    >>> adjusted_df = pd.DataFrame(
    ...     data={'numbers': []},
    ... )
    >>> drop_columns(df.copy(), adjusted_df, fields=['letters'])
       numbers
    0        0
    1        1
    2        2
    >>> drop_columns(df.copy(), adjusted_df, fields=['numbers'])
      letters  numbers
    0       a        0
    1       b        1
    2       c        2
    """
    if fields:
        try:
            missing_right = [f for f in fields if f != cols.name]
            if missing_right:
                df = df.drop(columns=missing_right)
        except AttributeError:
            missing_right = set(fields) - set(cols.columns)
            if missing_right:
                df = df.drop(columns=list(missing_right))
    return df


@enclose_string_fields
def override_cols(
        df: pd.DataFrame,
        cols: Union[pd.Series, pd.DataFrame],
        fields: Union[str, Iterable]
) -> pd.DataFrame:
    """
    Overrides column values of df with columns from cols if those columns
    exist in fields.

    Examples
    --------
    >>> df = pd.DataFrame({'letters': ['a', 'b', 'c'], 'numbers': [0, 1, 2]})
    >>> letters = pd.Series(
    ...     data=['w', 'x', 'y', 'z'],
    ...     name='letters'
    ... )
    >>> override_cols(df.copy(), letters, fields=['letters'])
      letters  numbers
    0       w        0
    1       x        1
    2       y        2
    >>> letters_and_nums = pd.DataFrame(
    ...     {'letters': ['w', 'x', 'y', 'z'], 'numbers': [22, 23, 24, 25]}
    ... )
    >>> override_cols(df.copy(), letters_and_nums, fields=['letters', 'numbers'])
      letters  numbers
    0       w       22
    1       x       23
    2       y       24
    >>> override_cols(df.copy(), letters_and_nums, fields=['letters'])
      letters  numbers
    0       w        0
    1       x        1
    2       y        2
    >>> override_cols(df.copy(), letters_and_nums, fields=['numbers'])
      letters  numbers
    0       a       22
    1       b       23
    2       c       24
    """
    if fields:
        try:
            existing_both = cols.name if cols.name in fields else None
            if existing_both:
                df[existing_both] = cols
        except AttributeError:
            existing_both = list(set(cols.columns) & set(fields))
            if existing_both:
                df[existing_both] = cols[existing_both]
    return df


@enclose_string_fields
def adjust_columns(
        df: pd.DataFrame,
        cols: Union[pd.Series, pd.DataFrame],
        fields: Union[str, Iterable]
) -> pd.DataFrame:
    """
    Adds, drops, and overrides columns added, dropped, or overridden by
    cols, an altered subset of df.

    Examples
    --------
    >>> df = pd.DataFrame({'letters': ['a', 'b', 'c'], 'numbers': [0, 1, 2]})
    >>> letters = pd.Series(
    ...     data=['w', 'x', 'y', 'z'],
    ...     name='letters'
    ... )
    >>> adjust_columns(df.copy(), letters, fields=['letters', 'numbers'])
      letters
    0       w
    1       x
    2       y
    >>> adjust_columns(df.copy(), letters, fields=['letters'])
      letters  numbers
    0       w        0
    1       x        1
    2       y        2

    >>> adjusted_df = pd.DataFrame(
    ...     {
    ...         'letters': ['w', 'x', 'y', 'z'],
    ...         'symbols': ['!', '@', '#', '$']
    ...     }
    ... )
    >>> adjust_columns(df.copy(), adjusted_df, fields=['letters', 'numbers'])
      letters symbols
    0       w       !
    1       x       @
    2       y       #
    >>> adjust_columns(df.copy(), adjusted_df, fields=['letters'])
      letters  numbers symbols
    0       w        0       !
    1       x        1       @
    2       y        2       #
    >>> adjust_columns(df.copy(), pd.DataFrame(), fields=['letters', 'numbers'])
    Empty DataFrame
    Columns: []
    Index: [0, 1, 2]
    """
    df = add_columns(df, cols, fields)
    df = drop_columns(df, cols, fields)
    df = override_cols(df, cols, fields)
    return df


def adjust_df(
        df: pd.DataFrame,
        cols: Union[pd.Series, pd.DataFrame],
        fields: Union[str, Iterable]
) -> pd.DataFrame:
    """
    Adjusts a dataframe to the contents of cols, a subset of dataframe that
    was altered.

    Rows that were added and dropped by the subset (cols) will be added and
    dropped by the dataframe (df) respectively.

    Columns that were added and dropped by the subset will be added and dropped
    by the dataframe respectively. The dataframe will then overwrite columns
    with the subset's columns with names that existed before and after the
    altercation.

    Parameters
    ----------
    df : pd.DataFrame
    cols : pd.DataFrame or pd.Series
        A subset of df that was altered.
    fields : str or Iterable
        The column name(s) of the subset before it was altered.

    Examples
    --------
    Changes in values of the subset overwrites the dataframe

    >>> df = pd.DataFrame({'letters': ['a', 'b', 'c'], 'numbers': [1, 2, 3]})
    >>> cols = pd.Series(['x', 'y' ,'z'], name='letters')
    >>> adjust_df(df.copy(), cols, fields=['letters'])
      letters  numbers
    0       x        1
    1       y        2
    2       z        3

    Adding rows from the subset adds rows to the dataframe.

    >>> cols = pd.Series(['a', 'b', 'c', 'd'], name='letters')
    >>> adjust_df(df.copy(), cols, fields=['letters'])
      letters  numbers
    0       a      1.0
    1       b      2.0
    2       c      3.0
    3       d      NaN

    Dropping rows from the subset drops rows from the dataframe.

    >>> cols = pd.Series(['a', 'b'], name='letters')
    >>> adjust_df(df.copy(), cols, fields=['letters'])
      letters  numbers
    0       a        1
    1       b        2
    >>> cols = pd.Series(['a', 'c'], name='letters')
    >>> adjust_df(df.copy(), cols, fields=['letters'])
      letters  numbers
    0       a        1
    1       c        2

    Adding columns in the subset adds columns to the dataframe.

    >>> data = {
    ...     'letters': ['x', 'y', 'z'],
    ...     'numbers': [1, 2, 3],
    ...     'symbols': ['!', '@', '#']
    ... }
    >>> cols = pd.DataFrame(data)
    >>> adjust_df(df.copy(), cols, fields=['letters', 'numbers'])
      letters  numbers symbols
    0       x        1       !
    1       y        2       @
    2       z        3       #

    Dropping columns from the subset drops columns from the dataframe.

    >>> cols = pd.Series(['a', 'b', 'c'], name='letters')
    >>> adjust_df(df.copy(), cols, fields=['letters', 'numbers'])
      letters
    0       a
    1       b
    2       c

    Empty subsets will result in the columns being dropped from the dataframe.

    >>> adjust_df(df.copy(), pd.DataFrame(), fields=['letters'])
       numbers
    0        1
    1        2
    2        3

    >>> adjust_df(df.copy(), pd.Series(), fields=['letters', 'numbers'])
    Empty DataFrame
    Columns: []
    Index: []
    """
    if cols.empty:
        df = df.drop(columns=fields)
        if df.empty:
            return pd.DataFrame()
    else:
        df = adjust_columns(df, cols, fields)
        df = adjust_rows(df, cols)
    return df
