from collections import MutableMapping, OrderedDict
from typing import Callable, Hashable, Mapping, Union, Iterable
import pandas as pd
from .adjust_df import adjust_df


def is_fields_in_col(fields: Union[str, Iterable], cols: Iterable) -> bool:
    """
    Returns true if all fields exist in cols. False otherwise.

    Parameters
    ----------
    fields : str or list-like
        Member check these field(s) against col.
    cols: list-like

    Returns
    --------
    bool
    """
    if isinstance(fields, str):
        if fields in cols:
            return True
        return False
    return all([True if f in cols else False for f in fields])


def _validate_key(k):
    """Validates if key is both an iterable and hashable."""
    try:
        iter(k)
        hash(tuple(k))
    except TypeError:
        msg = f"expected keys to be a string or a hashable " \
              f"iterable got: {k}"
        raise TypeError(msg)
    return k


def _validate_value(v):
    """Validates if value is a callable."""
    if not callable(v):
        msg = f"expected a callable, got type: {type(v)}"
        raise TypeError(msg)
    return v


def remove_duplicates(iterable):
    """Removes duplicates from an iterable."""
    return tuple(dict.fromkeys(iterable).keys())


class Standards(MutableMapping):
    """
    A mutable mapping of column names and standardization functions or
    callables for standardizing a pd.DataFrame instance.

    Methods
    -------
    standardize_df(df):
        Runs every column(s) with their corresponding callable if those
        column(s) are found in df, then return df. Re-adjusts the df if the
        return values from the callables have dropped or added rows and
        columns.
    """

    def __init__(
            self,
            d_: Mapping[Hashable, Callable] = None,
            remove_dupe: Union[Callable, None] = remove_duplicates, # for ordered removing of duplicates
            sort: Union[Callable, None] = sorted,
            as_single: bool = True,
            **kwargs
    ):
        """
        Constructor to create a Standards object.

        Arguments
        ---------
        d_: dict
            Mapping of field names (keys) and standardization callables (values).
        remove_dupe : callable, default set
            A callable to remove duplicates from keys that are iterables.
            Set to None to keep duplicates.
        sort : callable, default sorted
            A callable to sort keys that are iterables. Set to None if you
            don't want keys to be sorted.
        as_single : bool, default True
            If an iterable key only contains one item after duplication removal
            and sorting, then that item will be the key.
        **kwargs
            Mapping of field names (keys) and standardization callables (values).

        Examples
        --------
        The various ways iterable keys are affected by the following instance
        attributes: _remove_dupe, _sort, and _as_single.

        >>> import functools

        >>> def foo():
        ...     pass
        ...
        >>> kwargs = {
        ...     ('apple', 'apple'): foo,
        ...     ('oranges', 'pineapple', 'zebrapple'): foo,
        ...     'pear': foo
        ... }

        >>> str(Standards(kwargs)) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        "Standards([('apple', <function foo at ...>),
        (('oranges', 'pineapple', 'zebrapple'), <function foo at ...>),
        ('pear', <function foo at...>)])"

        >>> str(Standards(kwargs, remove_dupe=None)) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        "Standards([(('apple', 'apple'), <function foo at ...>),
        (('oranges', 'pineapple', 'zebrapple'), <function foo at ...>),
        ('pear', <function foo at...>)])"

        >>> new_sort = functools.partial(sorted, reverse=True)
        >>> str(Standards(kwargs, sort=new_sort)) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        "Standards([('apple', <function foo at ...>),
        (('zebrapple', 'pineapple', 'oranges'), <function foo at ...>),
         ('pear', <function foo at ...>)])"

        >>> str(Standards(kwargs, as_single=False)) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        "Standards([(('apple',), <function foo at ...>),
        (('oranges', 'pineapple', 'zebrapple'), <function foo at ...>),
        ('pear', <function foo at ...>)])"

        Notes
        -----
        See __setitem__ for implementation details.
        """

        self._remove_dupe = remove_dupe
        self._sort = sort
        self._as_single = as_single
        if d_ is None:
            d_ = {}
        d_.update(**kwargs)
        self._standards = OrderedDict()
        for k, v in d_.items():
            self.__setitem__(k, v)

    def __len__(self) -> int:
        return len(self._standards)

    def __getitem__(self, k: Hashable) -> Callable:
        return self._standards[k]

    def __setitem__(self, k: Hashable, v: Callable) -> None:
        """
        Assigns a key-value pair to self._standards.

        Duplicates items from iterable keys will be dropped then sorted.
        Afterwards, iterable keys with one item will have that one item be
        the key. These behaviors are overrideable (see Standard.__init__).

        Parameters
        ----------
        k : Hashable
            The key.
        v : Callable
            The value.

        Raises
        ------
        TypeError
            If o is not a callable, or if k is a hashable.
        """
        k = _validate_key(k)
        v = _validate_value(v)
        if isinstance(k, str):
            self._standards[k] = v
        else:
            if self._remove_dupe:
                k = self._remove_dupe(k)
            if self._sort:
                k = self._sort(k)
            k = tuple(k)
            if len(k) == 1 and self._as_single:
                self._standards[k[0]] = v
            else:
                self._standards[k] = v

    def __delitem__(self, k: Hashable):
        del self._standards[k]

    def __iter__(self) -> Iterable:
        return iter(self._standards)

    def __str__(self) -> str:
        """
        String representation of a Standards object for display.

        Returns
        -------
        str
            A string representing a Standards object.

        Examples
        --------
        >>> def format_amounts(amounts: pd.Series) -> pd.Series:
        ...     return amounts.apply(lambda x: '$ ' + str(x))
        ...
        >>> str(Standards(amounts=format_amounts)) # doctest: +ELLIPSIS
        "Standards([('amounts', <function format_amounts at...>)])"
        """
        cls_name = type(self).__name__
        components = [(k, v) for k, v in self._standards.items()]
        return f'{cls_name}({components})'

    def __repr__(self) -> str:
        """
        String representation of a Standards object for inspection.

        Returns
        -------
        str
            A string representing a Standards object.

        Examples
        --------
        >>> def format_amounts(amounts: pd.Series) -> pd.Series:
        ...     return amounts.apply(lambda x: '$ ' + str(x))
        ...
        >>> Standards(amounts=format_amounts) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Standards(remove_dupe=<function remove_duplicates at ...>,
        sort=<built-in function sorted>, as_single=True,
        standards=[('amounts', <function format_amounts at ...>)])
        """
        cls_name = type(self).__name__
        components = [(k, v) for k, v in self._standards.items()]
        cls_setting = 'remove_dupe={remove_dupe}, sort={sort}, ' \
                      'as_single={as_single}'
        cls_setting = cls_setting.format(
            remove_dupe=self._remove_dupe,
            sort=self._sort,
            as_single=self._as_single
        )
        return f'{cls_name}({cls_setting}, standards={components})'

    def standardize_df(
            self,
            df: pd.DataFrame,
            shared_only: bool = False,
            restrict_cols: bool = False,
            restrict_rows: bool = False,
    ) -> pd.DataFrame:
        """
        Standardizes a dataframe.

        Runs every columns (keys) with their corresponding callable (values)
        Runs df through every standardization callable (values) if all
        column names (key) can be found in df.columns.

        If column names is a single field, pass in df[field] (a series) to
        the standardization callable. If column names is an iterable of fields,
        then pass in df[list(fields)] (a dataframe) the the standardization
        callable.

        Re-adjusts the df if the return values from the callables have dropped
        or added rows and columns.

        Standardized subsets returned as an empty series or dataframe will
        result in the columns (associated to the callable) being dropped from
        the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to be standardized.

        shared_only: bool, default False
            With shared_only=False, standardize_df() raises a KeyError if there
            are fields from a standards instance missing field from the
            dataframe. Otherwise, ignore running standardization callables
            paired to field(s) missing from the dataframe.

            If restrict_cols=False, then callables associated with a missing
            field from the dataframe will be called if previous callables
            have created that column.

        restrict_cols: bool, default False
            With restrict_rows=True, standardized subsets cannot add or drop
            columns.

        restrict_rows: bool, default False
            With restrict_rows=True, standardized subsets cannot add or drop
            rows.

        Returns
        -------
        df : pd.DataFrame
            The standardized dataframe.

        Raises
        ------
        KeyError
            If shared_only=False and any fields from a standards instance are
            missing from the dataframe.
        ValueError
            If the standardized subset returned from the standardization
            callable has added or dropped any column(s) or row(s) when
            restrict_cols=True or restrict_rows=True respectively.

        Examples
        --------
        Single column operations.

        >>> data = {'letters': ['a', 'b', 'c'], 'numbers': [1, 2, 3]}
        >>> df = pd.DataFrame(data)
        >>> df
          letters  numbers
        0       a        1
        1       b        2
        2       c        3
        >>> def upper_letters(series):
        ...     return series.apply(lambda x: x.upper())
        ...
        >>> s = Standards(letters=upper_letters)
        >>> str(s) # doctest: +ELLIPSIS
        "Standards([('letters', <function upper_letters at ...>)])"
        >>> s.standardize_df(df)
          letters  numbers
        0       A        1
        1       B        2
        2       C        3

        Multi-column operations:

        >>> data = {'letters': ['a', 'b', 'c'], 'numbers': [1, 2, 3]}
        >>> df = pd.DataFrame(data)
        >>> def increment_one(dataframe):
        ...     "Increments numbers by one and letters by a character"
        ...     dataframe['numbers'] = dataframe['numbers'].apply(lambda x: x + 1)
        ...     dataframe['letters'] = dataframe['letters'].apply(
        ...         lambda x: chr(ord(x) + 1)
        ...     )
        ...     return dataframe
        ...
        >>> s = Standards()
        >>> s[('letters', 'numbers')] = increment_one
        >>> s.standardize_df(df)
          letters  numbers
        0       b        2
        1       c        3
        2       d        4

                Multiple operations

        >>> data = {'letters': ['a', 'b', 'c'], 'numbers': [1, 2, 3]}
        >>> df = pd.DataFrame(data)
        >>> df
          letters  numbers
        0       a        1
        1       b        2
        2       c        3
        >>> def add_symbols(dataframe):
        ...     '''adds a symbols column to the dataframe.'''
        ...     dataframe['symbols'] = ['!', '@', '#']
        ...     return dataframe
        ...
        >>> def increment_symbols(series):
        ...     '''increments symbols columns by 1 character.'''
        ...     return series.apply(lambda x: chr(ord(x) + 1))
        ...
        >>> s = Standards()
        >>> s[('letters', 'numbers')] = add_symbols
        >>> s['symbols'] = increment_symbols
        >>> s.standardize_df(df)
          letters  numbers symbols
        0       a        1       "
        1       b        2       A
        2       c        3       $

        Adding or dropping a row from a subset adds or drops the row from the
        dataframe respectively.

        >>> data = {'letters': ['a', 'b', 'c'], 'numbers': [1, 2, 3]}
        >>> df = pd.DataFrame(data)
        >>> def add_d(series):
        ...     return pd.Series(['a', 'b', 'c', 'd'], name='letters')
        ...
        >>> def drop_a(series):
        ...     return series[lambda x: x != 'a']
        ...
        >>> s = Standards()
        >>> s['letters'] = add_d
        >>> s.standardize_df(df)
          letters  numbers
        0       a      1.0
        1       b      2.0
        2       c      3.0
        3       d      NaN
        >>> s['letters'] = drop_a
        >>> s.standardize_df(df)
          letters  numbers
        1       b        2
        2       c        3

        With restrict_rows=True, standardized subsets cannot add or drop rows
        >>> def change_index(series):
        ...     series.index = [2, 3, 4]
        ...     return series
        ...
        >>> s['letters'] = add_d
        >>> s.standardize_df(df, restrict_rows=True) # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: lengths for leftover indexes and original dataframes
        indexes does not match, 4 != 3
        >>> s['letters'] = change_index
        >>> s.standardize_df(df, restrict_rows=True) # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: resulting indexes from standardization for fields letters
        does not match the original dataframes indexes.

        Adding or dropping a column from a subset adds or drops a column from
        the dataframe respectively.

        >>> def add_cols(dataframe):
        ...     dataframe['symbols'] = ['!', '@', '#']
        ...     return dataframe
        ...
        >>> def drop_cols(dataframe):
        ...     return dataframe['letters']
        ...
        >>> data = {'letters': ['a', 'b', 'c'], 'numbers': [1, 2, 3]}
        >>> df = pd.DataFrame(data)
        >>> s = Standards()
        >>> s[('letters', 'numbers')] = add_cols
        >>> s.standardize_df(df)
          letters  numbers symbols
        0       a        1       !
        1       b        2       @
        2       c        3       #
        >>> s[('letters', 'numbers')] = drop_cols
        >>> s.standardize_df(df)
          letters
        0       a
        1       b
        2       c

        With restrict_cols=True, standardized subsets cannot add or drop columns
        >>> s.standardize_df(df, restrict_cols=True) # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: expected columns '['letters', 'numbers']' ... got 'letters'

        Trying to standardize a dataframe missing any field from a standards
        instance results in a KeyError while shared_only=False (default value):
        >>> data = {'letters': ['a', 'b', 'c'], 'numbers': [1, 2, 3]}
        >>> df = pd.DataFrame(data)
        >>> def return_arg(arg):
        ...     return arg
        ...
        >>> s = Standards(
        ...     {'letters': return_arg, 'non_existing_field': return_arg}
        ... )
        >>> s.standardize_df(df, shared_only=False)
        Traceback (most recent call last):
            ...
        KeyError: "column 'non_existing_field' does not exist in the dataframe"

        With shared_only=True, standardize_df simply ignores running
        standardization callables paired to fields missing from the dataframe.
        >>> s.standardize_df(df, shared_only=True)
          letters  numbers
        0       a        1
        1       b        2
        2       c        3

        Note that if restrict_cols=False, then callables associated with a
        missing field from the dataframe will be called if previous operations
        have created that column, regardless if shared_only argument is true
        or false.

        >>> def add_symbols(series):
        ...     '''adds a symbols column to the dataframe.'''
        ...     data = {'letters': ['a', 'b', 'c'], 'symbols':['!', '@', '#']}
        ...     return pd.DataFrame(data)
        ...
        >>> def increment_symbols(series):
        ...     '''increments symbols columns by 1 character.'''
        ...     return series.apply(lambda x: chr(ord(x) + 1))
        ...
        >>> s = Standards()
        >>> s['letters'] = add_symbols
        >>> s['symbols'] = increment_symbols
        >>> s.standardize_df(df, shared_only=False)
          letters  numbers symbols
        0       a        1       "
        1       b        2       A
        2       c        3       $
        """
        df = df.copy()
        df_indexes = list(df.index)

        for n, (fields, fn) in enumerate(self._standards.items()):
            if not isinstance(fields, str):
                fields = list(fields)

            if not shared_only:
                msg = "column '{}' does not exist in the dataframe"
                if isinstance(fields, str):
                    if fields not in df.columns:
                        raise KeyError(msg.format(fields))
                else:
                    for f in fields:
                        if f not in df.columns:
                            raise KeyError(msg.format(f))
            else:
                if not is_fields_in_col(fields, df.columns):
                    continue

            leftover = fn(df.loc[:, fields].copy())

            if restrict_cols:
                try:
                    if leftover.name != fields:
                        msg = f"expected columns '{fields}' from operation " \
                              f"result, got '{leftover.name}'"
                        raise ValueError(msg)
                except AttributeError:
                    if list(leftover.columns) != list(fields):
                        msg = f"expected columns {list(fields)} from " \
                              f"operation result, got {list(leftover.columns)}"
                        raise ValueError(msg)

            if restrict_rows:
                if len(leftover) != len(df_indexes):
                    msg = f'lengths for leftover indexes and original ' \
                          f'dataframes indexes does not match, ' \
                          f'{len(leftover.index)} != {len(df_indexes)}'
                    raise ValueError(msg)
                if list(leftover.index) != df_indexes:
                    msg = f'resulting indexes from standardization for ' \
                          f'fields {fields} does not match the original ' \
                          f'dataframes indexes.'
                    raise ValueError(msg)

            df = adjust_df(df=df, fields=fields, cols=leftover)
        return df
