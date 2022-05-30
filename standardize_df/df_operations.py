from functools import singledispatch
import re
import numpy as np
import pandas as pd
from typing import \
    Any, \
    Callable, \
    Iterable, \
    Iterator, \
    Mapping, \
    MutableMapping, \
    Tuple, \
    Type, \
    Union
from .utils import \
    isempty, \
    remove_dup, \
    flatten_iterable as flat

Columns = Union[pd.Series, pd.DataFrame]


def convert_items(
        iterable: Iterable,
        type_: Type,
        ignore: Union[Type, Tuple] = None
) -> Iterator:
    """
    Type casts every item in an iterable to type_.

    If the attempt raises a ValueError or TypeError, or the item is of type(s)
    ignore will be yielded  as is.

    iterable : iterable
        Items to be type casted.
    type_: type
        Used for type casting every item in iterable. Cannot be a type
        from the ignore argument.
    ignore : type or tuple of types, optional
        Items of this type(s) will simply be yielded without attempts to type
        cast it.

    Yields
    ------
    any

    Raises
    ------
    ValueError
        If type_ is in ignore.

    Examples
    --------
    >>> converted = convert_items(['one', '2', '3'], type_=int)
    >>> list(converted)
    ['one', 2, 3]
    >>> converted = convert_items(['1', ['2', 3]], type_=tuple, ignore=str)
    >>> list(converted)
    ['1', ('2', 3)]
    """
    try:
        if type_ in ignore:
            raise ValueError(f'Type {type_} cannot be a type from ignore.')
    except TypeError:
        if type_ == ignore:
            raise ValueError(f'Type {type_} cannot be a type from ignore.')
    for i in iterable:
        try:
            if ignore and isinstance(i, ignore):
                yield i
            else:
                yield type_(i)
        except (ValueError, TypeError):
            yield i


def agg_unique_to_tuple(
        df: pd.DataFrame,
        agg_on: Iterable,
        flatten: bool = False
) -> pd.Series:
    """
    Aggregates multiple entries into a single entry, combining fields in agg_on
    into a tuple if it has multiple unique values.

    Arguments
    ---------
    df : pd.DataFrame
        A grouped dataframe member.
    agg_on : Iterable
        An iterable of column names existing in df.
    flatten : bool, default False
        Flattens all values from a column before unique values are determined.
    Returns
    -------
    pd.Series
        A single aggregated entry with tuple's for fields in agg_on that had
        multiple unique values.

    Examples
    --------
    >>> data = {
    ...     'name': ['John', 'John'],
    ...     'id': [123, 123],
    ...     'favorite_food': [('chicken', 'dog'), 'cow'],
    ...     'favorite_pet': ['chicken', ('dog', 'cat')]
    ... }
    >>> grouped_john = pd.DataFrame(data)
    >>> john = agg_unique_to_tuple(
    ...     grouped_john,
    ...     agg_on=['favorite_food']
    ... )
    >>> john
    name                              John
    id                                 123
    favorite_food    ((chicken, dog), cow)
    favorite_pet                   chicken
    Name: 0, dtype: object
    >>> john = agg_unique_to_tuple(
    ...     df=grouped_john,
    ...     agg_on=['favorite_food', 'favorite_pet']
    ... )
    >>> john
    name                              John
    id                                 123
    favorite_food    ((chicken, dog), cow)
    favorite_pet     (chicken, (dog, cat))
    Name: 0, dtype: object
    >>> john = agg_unique_to_tuple(
    ...     df=grouped_john,
    ...     agg_on=['favorite_food', 'favorite_pet'],
    ...     flatten=True
    ... )
    >>> john
    name                            John
    id                               123
    favorite_food    (chicken, dog, cow)
    favorite_pet     (chicken, dog, cat)
    Name: 0, dtype: object
    >>> data = {
    ...     'name': ['John', 'John'],
    ...     'id': [123, 123],
    ...     'favorite_food': [('chicken', 'dog'), 'cow'],
    ...     'favorite_pet': ['chicken', ('dog', 'cat')]
    ... }
    """
    row = df.iloc[0, :]
    for col in agg_on:
        if flatten:
            values = flat(df[col].to_list())
        else:
            values = df[col].to_list()
        values = convert_items(values, tuple, ignore=str)
        # Get uniques while retaining order.
        unique_values = list(dict.fromkeys(values))
        col_data = tuple(i for i in unique_values if not isempty(i))
        if len(col_data) > 1:
            row[col] = col_data
    return row


@singledispatch
def _convert_datetime(
        cols: pd.Series,
        date_format: str,
        infer_datetime_format: bool = False
) -> pd.Series:
    return cols.apply(
        pd.to_datetime,
        format=date_format,
        infer_datetime_format=infer_datetime_format
    )


@_convert_datetime.register
def _(
        cols: pd.DataFrame,
        date_format: str,
        infer_datetime_format: bool = False
) -> pd.DataFrame:
    for c in cols.columns:
        cols[c] = cols[c].apply(
            pd.to_datetime,
            format=date_format,
            infer_datetime_format=infer_datetime_format
        )
    return cols


def convert_datetime(
        cols: Union[pd.Series, pd.DataFrame],
        date_format: str,
        infer_datetime_format: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Converts the dtype of a series or dataframe to datetime if date_format
    is provided.

    Note that for dates that don't match with the date format, an attempt to
    infer the correct date format will be made if infer_datetime_format=True.

    Parameters
    ----------
    cols: pd.Series or pd.DataFrame
        The column(s) to convert.
    date_format: str
        The strftime to parse time, eg "%d/%m/%Y".
    infer_datetime_format : bool, default False
        From pandas.to_datetime docs (Version: 1.1.5):
        "If True and no format is given, attempt to infer the format of the
        datetime strings based on the first non-NaN element, and if it can be
        inferred, switch to a faster method of parsing them. In some cases this
        can increase the parsing speed by ~5-10x."

    Returns
    -------
    pd.Series or pd.DataFrame
        Column(s) with its date values formatted.

    See Also
    --------
    pd.to_datetime

    Examples
    --------
    Converting a series

    >>> dates = pd.Series(['03/01/2000', '03/02/2000', '03/03/2000'])
    >>> convert_datetime(dates, date_format='%d/%m/%Y')
    0   2000-01-03
    1   2000-02-03
    2   2000-03-03
    dtype: datetime64[ns]
    >>> convert_datetime(dates, date_format='%m/%d/%Y')
    0   2000-03-01
    1   2000-03-02
    2   2000-03-03
    dtype: datetime64[ns]

    Converting a dataframe

    >>> dates = pd.DataFrame(
    ...     {
    ...         'mar': ['03/01/2000', '03/02/2000', '03/03/2000'],
    ...         'apr': ['04/01/2000', '04/02/2000', '04/03/2000']
    ...     }
    ... )
    >>> convert_datetime(dates, date_format='%m/%d/%Y')
             mar        apr
    0 2000-03-01 2000-04-01
    1 2000-03-02 2000-04-02
    2 2000-03-03 2000-04-03

    If infer_datetime_format=True, the function attempts to infer datetime
    format rows not matching with the given date_format.

    >>> dates = pd.Series(['03/30/2022', '03/01/2022', '03/04/2022'])
    >>> convert_datetime(
    ...     dates,
    ...     date_format='%d/%m/%Y',
    ...     infer_datetime_format=True
    ... )
    0   2022-03-30
    1   2022-01-03
    2   2022-04-03
    dtype: datetime64[ns]

    See Also
    --------
    pd.to_datetime
    """
    if date_format or infer_datetime_format:
        return _convert_datetime(
            cols,
            date_format,
            infer_datetime_format=infer_datetime_format
        )
    return cols


@singledispatch
def _convert_dtype(cols: pd.Series, dtype) -> pd.Series:
    return cols.astype(dtype=dtype)


@_convert_dtype.register
def _(cols: pd.DataFrame, dtype) -> pd.DataFrame:
    try:
        columns = [k for k in dtype.keys()]
        for c in columns:
            cols[c] = cols[c].astype(dtype=dtype[c])
    except AttributeError:
        columns = cols.columns
        for c in columns:
            cols[c] = cols[c].astype(dtype=dtype)
    return cols


def convert_dtype(cols: Columns, dtype=None) -> Columns:
    """
    Converts the dtype of a series or dataframe to datetime.

    cols: pd.Series or pd.DataFrame
        The column(s) to convert.
    dtype: data type, or dict of column name -> data type
        If cols is a pd.Series, Use a numpy.dtype or Python type to cast entire
        pandas object to the same type. For a pd.DataFrame, optionally use
        {col: dtype, ...}, where col is a column label and dtype is a
        numpy.dtype or Python type to cast one or more of the DataFrame's
        columns to column-specific types.

    Results
    -------
    pd.Series or pd.DataFrame
        If dtype is not a callable, return the converted series or dataframe,
        whichever type cols was.

    Examples
    --------

    Convert a series

    >>> numbers = pd.Series([1, 2, 3])
    >>> numbers
    0    1
    1    2
    2    3
    dtype: int64
    >>> convert_dtype(numbers, 'string')
    0    1
    1    2
    2    3
    dtype: string

    Convert all columns of a dataframe

    >>> numbers = pd.DataFrame({'ints':[1, 2, 3], 'floats': [1.0, 2.0, 3.0]})
    >>> numbers
       ints  floats
    0     1     1.0
    1     2     2.0
    2     3     3.0
    >>> numbers['ints'].dtype, numbers['floats'].dtype
    (dtype('int64'), dtype('float64'))
    >>> convert_dtype(numbers, 'string')
      ints floats
    0    1    1.0
    1    2    2.0
    2    3    3.0
    >>> numbers['ints'].dtype, numbers['floats'].dtype
    (string[python], string[python])

    Convert specific columns with a dict

    >>> convert_dtype(numbers, {'ints': 'float', 'floats': 'string'})
       ints floats
    0   1.0    1.0
    1   2.0    2.0
    2   3.0    3.0
    >>> numbers['ints'].dtype, numbers['floats'].dtype
    (dtype('float64'), string[python])

    See Also
    --------
    pd.Series.astype
    """
    if dtype:
        return _convert_dtype(cols, dtype)
    return cols


def drop_columns(df: pd.DataFrame, drop: Iterable) -> pd.DataFrame:
    """
    Drop columns from a dataframe.

    Arguments
    ---------
    df : pd.DataFrame
    drop : Iterable of column names and Iterable of column names
        The columns to be dropped from df. The iterable can be composed of
        nested iterable of column names. Ex: ['col1', ['col2', 'col3']]

    Returns
    -------
    pd.DataFrame
        The pd.DataFrame with dropped columns.

    Examples
    --------
    >>> data = {
    ...     'numbers': [1, 2, 3],
    ...     'letters': ['a', 'b', 'c'],
    ...     'symbols': ['!', '@', '#']
    ... }
    >>> df = pd.DataFrame(data)
    >>> drop_columns(df, ['numbers'])
      letters symbols
    0       a       !
    1       b       @
    2       c       #
    >>> drop_columns(df, ['letters', ['numbers']])
      symbols
    0       !
    1       @
    2       #

    See Also
    --------
    pd.DataFrame.drop
    """
    return df.drop(
        columns=[i for i in df.columns if i in flat(drop)]
    )


def fill_columns(
        df: pd.DataFrame,
        fields: Iterable,
        empty: Union[MutableMapping, Any] = np.nan
) -> pd.DataFrame:
    """
    Add empty columns from fields to a DataFrame.

    df : pd.DataFrame
    fields : bool, default False
        Add fields of empty values to the DataFrame.
    empty : Mutable Mapping or Any, default np.nan
        Fills column with this value. If given a mapping, fills empty fields
        (keys) with the paired value.

    Returns
    -------
    df : pd.DataFrame
        Standardized dataframe.

    Examples
    --------
    >>> data = {
    ...     'numbers': [1, 2, 3],
    ...     'letters': ['a', 'b', 'c']
    ... }
    >>> df = pd.DataFrame(data)
    >>> fill_columns(df.copy(), ['symbols'])
       numbers letters  symbols
    0        1       a      NaN
    1        2       b      NaN
    2        3       c      NaN
    >>> from decimal import Decimal
    >>> filled = fill_columns(
    ...     df.copy(),
    ...     ['symbols', 'decimals'],
    ...     {'symbols': '', 'decimals': Decimal()}
    ... )
    >>> filled
       numbers letters symbols decimals
    0        1       a                0
    1        2       b                0
    2        3       c                0
    >>> filled['decimals'].iloc[0]
    Decimal('0')
    """
    missing = [f for f in fields if f not in df.columns]
    try:
        for field in missing:
            df[field] = empty[field]
    except TypeError:
        for field in missing:
            df[field] = empty
    return df


def validate_filter_args(
        keep: Iterable,
        drop: Iterable
) -> Tuple[Iterable, Iterable]:
    """ Validates that there are no shared items in keep and drop. """
    if keep and drop:
        both = filter(lambda k: k in drop, keep)
        for b in both:
            msg = f"keep's filter keywords '{b}' in drop: {drop}"
            raise ValueError(msg)
    return keep, drop


@singledispatch
def _filter(
        cols: pd.Series,
        keep: Iterable = None,
        drop: Iterable = None
) -> pd.Series:
    if keep:
        return cols[cols.isin(keep)]
    if drop:
        return cols[~cols.isin(drop)]
    return cols


@_filter.register
def _(
        cols: pd.DataFrame,
        keep: Iterable = None,
        drop: Iterable = None
) -> pd.DataFrame:
    if keep:
        return cols[cols.isin(keep).any(axis=1)]
    if drop:
        return cols[~cols.isin(drop).any(axis=1)]
    return cols


def filter_(
        cols: Columns,
        keep: Iterable = None,
        drop: Iterable = None,
        fn: Callable = None
) -> Columns:
    """
    Filter out values in filter from column.

    Parameters
    ----------
    cols : pd.Series or pd.DataFrame
        A pandas column or dataframe

    keep : Iterable
        Values to keep from the column(s).
    drop : Iterable
        Values to drop from the column(s).
    fn : Callable
        If provided, overrides this operation. Returns fn(cols).
    Returns
    -------
    cols : pd.Series or pd.DataFrame
        Filtered column(s).

    Examples
    --------

    Filtering a series

    >>> fruits = pd.Series(['apple', 'pineapple', 'orange'], name='fruits')
    >>> filter_(fruits, keep=['apple'])
    0    apple
    Name: fruits, dtype: object
    >>> filter_(fruits, drop=['apple'])
    1    pineapple
    2       orange
    Name: fruits, dtype: object
    >>> filter_(fruits, keep=['apple', 'pineapple'], drop=['orange'])
    0        apple
    1    pineapple
    Name: fruits, dtype: object

    Filtering a dataframe.

    >>> foods = pd.DataFrame(
    ...     {
    ...         'fruits': ['apple', 'pineapple', 'orange'],
    ...         'vegetables': ['celery', 'broccoli', 'spinach']
    ...     }
    ... )
    >>> filter_(foods, keep=['apple', 'celery', 'broccoli'])
          fruits vegetables
    0      apple     celery
    1  pineapple   broccoli
    >>> filter_(foods, drop=['apple', 'spinach'])
          fruits vegetables
    1  pineapple   broccoli

    Cannot keep and drop the same value.

    >>> filter_(fruits, keep=['apple'], drop=['apple', 'pineapple'])
    Traceback (most recent call last):
        ...
    ValueError: keep's filter keywords 'apple' in drop: ['apple', 'pineapple']
    """
    try:
        return fn(cols)
    except TypeError:
        keep, drop = validate_filter_args(keep, drop)
        return _filter(cols, keep=keep, drop=drop)


def group_aggregate_unique(
        df: pd.DataFrame,
        groupby: Iterable,
        agg_on: Iterable,
        flatten: bool = False
) -> pd.DataFrame:
    """
    Groups a dataframe then aggregates each group into a single entry. Combine
    each group's column into a tuple if it contain multiple unique values.

    Arguments
    ---------
    df : pd.DataFrame
        Dataframe to group and aggregate.
    groupby : Iterable
        Column names to group the dataframe by
    agg_on : Iterable
        Column names to aggregate a group's column into a tuple if it
        contain multiple unique values.
    flatten : bool, default False
        Flattens all values from a column before unique values are determined.
    Returns
    -------
    pd.DataFrame
        An aggregated dataframe.

    Examples
    --------
    >>> data = {
    ...     'fruits': ['apple', 'apple'],
    ...     'vegetables': ['celery', ['celery', 'spinach']],
    ...     'drinks': [['water', 'black tea'], 'green tea']
    ... }
    >>> foods = pd.DataFrame(data)
    >>> group_aggregate(foods, groupby=['fruits'], agg_on=['vegetables'])
      fruits                   vegetables              drinks
    0  apple  (celery, (celery, spinach))  [water, black tea]
    >>> group_aggregate(
    ...     foods,
    ...     groupby=['fruits'],
    ...     agg_on=['vegetables'],
    ...     flatten=True
    ... )
      fruits         vegetables              drinks
    0  apple  (celery, spinach)  [water, black tea]
    >>> group_aggregate(
    ...     foods,
    ...     groupby=['fruits'],
    ...     agg_on=['vegetables', 'drinks'],
    ...     flatten=True
    ... )
      fruits         vegetables                         drinks
    0  apple  (celery, spinach)  (water, black tea, green tea)
    """
    df = df.groupby(
        by=groupby,
        as_index=False,
        dropna=False
    ).apply(lambda df: agg_unique_to_tuple(df, agg_on=agg_on, flatten=flatten))
    if len(df) == 1:  # if only 1 value, df col name becomes first items idx
        df.columns.name = None
    return df


group_aggregate = group_aggregate_unique


def format_(cols: Columns, format: Callable) -> Columns:
    """
    Format a column; pass in 'cols' to 'format' and return the results.

    Arguments
    ---------
    cols : pd.Series or pd.DataFrame
        Column to pass into format
    format : Callable
        A callable to format cols.

    Returns
    -------
    A formatted column(s).
    """
    try:
        return format(cols)
    except TypeError:
        return cols


def merge_left(
        left: pd.DataFrame,
        right: pd.DataFrame,
        merge_on: Iterable
) -> pd.DataFrame:
    """
    A left merge.

    From pandas.merge docstring (v. 1.14.4):
    "Use only keys from left frame, similar to a SQL left outer join;
    preserve key order."

    Arguments
    ---------
    left : pd.DataFrame
        The left frame. The merge will only use keys from this frame.
    right : pd.DataFrame
        The right frame. This operation will only merge keys from the left
        frame.
    merge_on :
        From pandas.merge docstring (v. 1.14.4):
        "Column or index level names to join on. These must be found in both
        DataFrames. If on is None and not merging on indexes then this defaults
        to the intersection of the columns in both DataFrames."
    Returns
    -------
    pd.DataFrame
        Merged dataframe.

    Examples
    --------
    >>> df1 = pd.DataFrame({'nums': [1, 3], 'letters': ['a', 'b']})
    >>> df2 = pd.DataFrame({'nums': [1, 2], 'symbols': ['!', '@']})
    >>> merge_left(df1, df2, merge_on=['nums'])
       nums letters symbols
    0     1       a       !
    1     3       b     NaN
    >>> merge_left(df2, df1, merge_on=['nums'])
       nums symbols letters
    0     1       !       a
    1     2       @     NaN
    """
    right = right.drop_duplicates(subset=merge_on)
    df = pd.merge(
        left=left,
        right=right,
        how='left',
        on=merge_on,
        suffixes=('', 'drop'),
    )
    columns = [field for field in df.columns if
               field.endswith('drop')]
    df = df.drop(columns=columns)
    return df


def order_columns(df: pd.DataFrame, prio: Iterable) -> pd.DataFrame:
    """
    Sort dataframe columns by prio, then by any other columns in df.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be ordered.
    prio: list-like
        Prioritize these columns before any other columns in df for the sort.
        Columns missing will be ignored.

    Returns
    --------
    pd.DataFrame
        Dataframe with sorted columns.

    Examples
    --------
    >>> df = pd.DataFrame({'letters': [], 'numbers': [], 'symbols':[]})
    >>> order_columns(df, prio=['symbols', 'letters']).columns
    Index(['symbols', 'letters', 'numbers'], dtype='object')
    >>> order_columns(df, prio=['numbers']).columns
    Index(['numbers', 'letters', 'symbols'], dtype='object')
    """
    def sort_by_priority():
        priority = tuple(col for col in prio if col in df.columns)
        for p in priority:
            yield p
        for col in df.columns:
            if col not in priority:
                yield col
    return df[list(sort_by_priority())]


def rename_columns(
        df: pd.DataFrame,
        mapping: MutableMapping,
        strip_names: bool = False
) -> pd.DataFrame:
    """
    Rename column names.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to rename.
    mapping : Mapping {str:str}, default None
        From pandas.DataFrame.rename docstrings (v. 1.14.4):
        "Rename the existing column name (keys) to the intended (values)."

    strip_names :  bool, default True
        Strips all leading and trailing whitespace from every column name in
        the mapping (keys and values) and dataframe before renaming the
        dataframe.

    Returns
    --------
    pd.DataFrame
        Dataframe with sorted columns.

    See Also
    --------
    pd.DataFrame.rename

    Examples
    --------
    >>> df = pd.DataFrame({'lets': [], 'nums': [], 'syms':[]})
    >>> rename = {'lets': 'letters', 'nums': 'numbers', 'syms': 'symbols'}
    >>> rename_columns(df, rename).columns
    Index(['letters', 'numbers', 'symbols'], dtype='object')

    With strip_names=True
    >>> df = pd.DataFrame({'  lets': [], 'nums  ': [], '  syms  ':[]})
    >>> rename = {
    ...     'lets': '  letters',
    ...     'nums': 'numbers  ',
    ...     'syms': '  symbols  '
    ... }
    >>> rename_columns(df, rename, strip_names=True).columns
    Index(['letters', 'numbers', 'symbols'], dtype='object')
    """
    if strip_names:
        columns = {col: col.strip() for col in df.columns}
        df = df.rename(columns=columns)
        mapping = {k.strip(): v.strip() for k, v in mapping.items()}
    if mapping:
        return df.rename(columns=mapping)


@singledispatch
def _replace_nulls(cols: pd.Series, replace: Any = None) -> pd.Series:
    cols = cols.fillna(replace)
    return cols


@_replace_nulls.register
def _(cols: pd.DataFrame, replace: Any = None) -> pd.DataFrame:
    for c in cols.columns:
        cols[c] = cols[c].fillna(replace)
    return cols


def replace_nulls(
        cols: Columns,
        replace: Union[Callable, Any] = None
) -> Columns:
    """
    Fill column null values with replace_nulls.

    Parameters
    ----------
    cols : pd.Series or pd.DataFrame
        A pandas column or dataframe
    replace : callable or any
        The replace value of null values in cols. If it's a callable, return
        replace(cols).

    Returns
    -------
    pd.Series
        A pandas column with its null values replaced.

    See Also
    --------
    pd.Series.fillna

    Examples
    --------
    >>> fruits = pd.Series(['apple', 'pineapple', np.nan], name='fruits')
    >>> replace_nulls(fruits, 'unknown fruit')
    0            apple
    1        pineapple
    2    unknown fruit
    Name: fruits, dtype: object
    >>> data = {
    ...     'fruits': ['apple', 'pineapple', np.nan],
    ...     'vegetables': [np.nan, 'broccoli', 'spinach']
    ... }
    >>> foods = pd.DataFrame(data)
    >>> replace_nulls(foods, 'unknown food')
             fruits    vegetables
    0         apple  unknown food
    1     pineapple      broccoli
    2  unknown food       spinach
    """
    if replace is not None:
        try:
            return replace(cols)
        except TypeError:
            return _replace_nulls(cols, replace=replace)
    return cols


def standardize_columns(
        df: pd.DataFrame,
        shape_to: Iterable = None,
        mapping: MutableMapping = None,
        strip_names: bool = True,
        drop_cols: bool = True,
        fill: bool = True,
        ordered: bool = True
) -> pd.DataFrame:
    """
    Standardizes a dataframe's columns.

    Rename the dataframe and "shapes" (add, drop, or re-order) columns
    according to shape_to.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas series or dataframe.
    shape_to : iterable of column names
        Column names to "shape" (add, drop, or re-order) the dataframe.

        Any columns from shape_to missing in df will be added to df. Any
        columns from df missing in shape_to will be dropped from the df.
        The dataframe will be re-ordered according to the order of shape_to.

        Disable certain functionalities of shaping a dataframe by setting the
        following arguments to False: fill, drop_cols, and ordered.
    mapping : Mapping
        Rename columns with a mapping composed of current column name (keys)
        to converted column names (values).
    strip_names : bool, default True
        If True, field names from the mapping (keys and values) and dataframe
        will be stripped of any trailing and leading whitespaces.
    drop_cols : bool, default True
        If True, columns in the dataframe not in fields of shape_to will be
        dropped from the dataframe.
    fill : bool, default True
        If True, columns from shape_to missing the dataframe will be added to
        the dataframe.
    ordered : bool, default True
        If True, the dataframe will order its columns to match the fields in
        shape_to. Columns not found in shape_to will be added to the end
        of the dataframe, ordering left (front) to right (back) by their
        position in the original dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with standardized columns.

    Examples
    --------

    The dataframe is "shaped" according to shape_to.

    >>> data = {
    ...     'fruits': ['apple', 'pineapple', 'orange'],
    ...     'vegetables': ['celery', 'broccoli', 'spinach']
    ... }
    >>> foods = pd.DataFrame(data)
    >>> standardize_columns(foods, shape_to=['junk_food', 'fruits'])
       junk_food     fruits
    0        NaN      apple
    1        NaN  pineapple
    2        NaN     orange

    Disable certain functionalities of "shaping" the dataframe with the
    following arguments: fill, drop_cols, and ordered.

    >>> standardize_columns(foods, shape_to=['junk_food', 'fruits'], fill=False)
          fruits
    0      apple
    1  pineapple
    2     orange

    The dataframe will be renamed, if given a mapping arg, before its columns
    are "shaped".

    >>> standardize_columns(
    ...     foods,
    ...     shape_to=['junk_food', 'fruits'],
    ...     mapping={'fruits': 'fruit_loops'},
    ...     drop_cols=False
    ... )
       junk_food  fruits fruit_loops vegetables
    0        NaN     NaN       apple     celery
    1        NaN     NaN   pineapple   broccoli
    2        NaN     NaN      orange    spinach
    """
    if mapping:
        df = rename_columns(
            df,
            mapping=mapping,
            strip_names=strip_names
        )

    if shape_to:
        shape_to = remove_dup(flat(shape_to))
        if drop_cols:
            drop_fields = [i for i in df.columns if i not in shape_to]
            df = drop_columns(df, drop=drop_fields)

        if fill:
            df = fill_columns(df, fields=shape_to)

        if ordered:
            df = order_columns(df, prio=shape_to)

    return df


def is_mixed_mapping(mapping: Mapping, dtype: type) -> bool:
    """
    Return true if all keys and values is of type dtype

    Parameters
    ----------
    mapping: dict
        Mapping to compare it's keys and values to dtype
    dtype: type
        Data type to compare all Mapping keys and values.
    """
    return not all([isinstance(k, dtype) and isinstance(v, dtype) for
                    k, v in mapping.items()])


def mixed_replace(
        col: pd.Series,
        replacements: Mapping[Any, Any],
        case: bool = True,
        regex: bool = False
) -> pd.Series:
    """
    Replaces values matching the target pattern (keys) with the replacement
    value (values) in a column. Replacement values can be of mixed data types.
    Case insensitive replacements and/or regex replacements is supported.

    Unlike the behaviors of replace methods from a pd.Series, pd.DataFrame,
    or pd.Series.str, if parts of a value matches the target pattern, the
    entire value will be replaced. Not parts of the value.


    Parameters
    ----------
    col : pandas.Series
        A pandas column.
    replacements: mapping
        Mapping of pattern (keys) and its replacement (values).
    case: bool, default True
        Replacements are case sensitive if True, case insensitive otherwise.
        Will only apply to pairs that are both strings.
    regex: bool, default False
        Replace using regex if True. Keys will be escaped if False.
        Will only apply to pairs that are both strings.

    Returns
    -------
    pandas.Series
        A pandas column with replaced values.
    """
    for k, v in replacements.items():
        if isinstance(k, str):
            col = str_replace(
                col=col,
                replacements={k: v},
                case=case,
                regex=regex,
            )
        else:
            col = col.replace(k, v)
    return col


def str_replace(
        col: pd.Series,
        replacements: Mapping[str, str] = None,
        case: bool = True,
        regex: bool = False
):
    """
    Replaces values matching the target pattern (keys) with the replacement
    value (values) in a column for string only replacements (keys and values).
    Case insensitive replacements and/or regex replacements is supported.

    Unlike the behaviors of replace methods from a pd.Series, pd.DataFrame,
    or pd.Series.str, if parts of a value matches the target pattern, the
    entire value will be replaced. Not parts of the value.

    Parameters
    ----------
    col : pandas.Series
        A pandas column.
    replacements: mapping, optional
        Mapping of pattern (keys) and its replacement (values).
    case: bool, default True
        replacements are case sensitive if True, or case insensitive if False.
        Will only apply to pairs that are both strings.
    regex: bool, default False
        Replace using regex if True. Keys will be escaped if False.
        Will only apply to pairs that are both strings.

    Returns
    -------
    pandas.Series
        A pandas column with replaced values.
    """
    if not regex:
        replacements = {re.escape(k): v for k, v in replacements.items()}
    for k, v in replacements.items():
        repl_idx = np.where(
            col.str.fullmatch(k, case=case) & col.notnull()
        )
        repl_idx = repl_idx[0]
        try:
            col.iloc[repl_idx] = v
        except ValueError:
            col = col.astype(object)  # can still access series.str manager.
            col.iloc[repl_idx] = v
    return col


def replace_series(
        col: pd.Series,
        replacements: Mapping[Any, Any],
        case: bool = True,
        regex: bool = False
):
    """
    Replaces values matching the target pattern (keys) with the replacement
    value (values) in a column. Case insensitive replacements and/or
    regex replacements is supported.

    Unlike the behaviors of replace methods from a pd.Series, pd.DataFrame,
    or pd.Series.str, if parts of a value matches the target pattern, the
    entire value will be replaced. Not parts of the value.

    Parameters
    ----------
    col : pandas.Series
        A pandas column.
    replacements: mapping
        Mapping of pattern (keys) and its replacement (values).
    case: bool, default True
        replacements are case sensitive if True, or case insensitive if False.
        Will only apply to pairs that are both strings.
    regex: bool, default False
        Replace using regex if True. Keys will be escaped if False.
        Will only apply to pairs that are both strings.

    Returns
    -------
    pandas.Series
        A pandas column with replaced values.

    Examples
    --------
    >>> replacements = {'a':'1', 'B':'2', '(c)': '3'}
    >>> letters = pd.Series(['a', 'b', '(C)'], name='letters')
    >>> letters.head()
    0      a
    1      b
    2    (C)
    Name: letters, dtype: object

    Case sensitive and no regex replacements

    >>> replace_series(letters, replacements=replacements)
    0      1
    1      b
    2    (C)
    Name: letters, dtype: object

    Case insensitive and no regex replacements

    >>> replace_series(letters, replacements=replacements,case=False)
    0    1
    1    2
    2    3
    Name: letters, dtype: object

    Case insensitive and regex replacements

    >>> replace_series(letters, replacements=replacements, case=False, regex=True)
    0    1
    1    2
    2    3
    Name: letters, dtype: object
    """
    is_mixed = is_mixed_mapping(replacements, dtype=str)
    if is_mixed:
        col = mixed_replace(
            col,
            replacements=replacements,
            case=case,
            regex=regex,
        )
    else:
        col = str_replace(
            col,
            replacements=replacements,
            case=case,
            regex=regex,
        )
    return col


def replace_df(
        df: pd.DataFrame,
        replacements: Mapping[Any, Any],
        case: bool = True,
        regex: bool = False
) -> pd.DataFrame:
    """
    Replaces values matching the target pattern (keys) with the replacement
    value (values) in all columns from the dataframe. Case insensitive
    replacements and/or regex replacements is supported.

    Unlike the behaviors of replace methods from a pd.Series, pd.DataFrame,
    or pd.Series.str, if parts of a value matches the target pattern, the
    entire value will be replaced. Not parts of the value.

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas dataframe to replace its values.
    replacements: mapping
        Mapping of pattern (keys) and its replacement (values).
    case: bool, default True
        replacements are case sensitive if True, or case insensitive if False.
        Will only apply to pairs that are both strings.
    regex: bool, default False
        Replace using regex if True. Keys will be escaped if False.
        Will only apply to pairs that are both strings.

    Returns
    -------
    pandas.DataFrame
        A pandas dataframe with replaced values.

    Examples
    --------
    >>> replacements = {
    ...     'golden retriever': 'goldie boy',
    ...     'dove': 'lovey dovey',
    ...     '.*pug.*': 'dog that cannot breath'
    ... }
    >>> animals = pd.DataFrame({
    ...     'dogs': ['poodle', '(pug)', 'golden retriever'],
    ...     'birds': ['Pigeon', 'Crow', 'Dove']
    ... })
    >>> animals.head()
                   dogs   birds
    0            poodle  Pigeon
    1             (pug)    Crow
    2  golden retriever    Dove

    Case sensitive and no regex replacements

    >>> replace_df(animals, replacements=replacements)
             dogs   birds
    0      poodle  Pigeon
    1       (pug)    Crow
    2  goldie boy    Dove

    Case insensitive and no regex replacements

    >>> replace_df(animals, replacements=replacements, case=False)
             dogs        birds
    0      poodle       Pigeon
    1       (pug)         Crow
    2  goldie boy  lovey dovey

    Case insensitive and regex replacements
    >>> replace_df(animals, replacements=replacements, case=False, regex=True)
                         dogs        birds
    0                  poodle       Pigeon
    1  dog that cannot breath         Crow
    2              goldie boy  lovey dovey
    """
    for c in df.columns:
        df[c] = replace_series(
            df[c],
            replacements=replacements,
            case=case,
            regex=regex
        )
    return df


def replace(
        cols: Columns,
        replacements: Mapping[Any, Any] = None,
        case: bool = True,
        regex: bool = False
) -> Columns:
    """
    Replaces values matching the target pattern (keys) with the replacement
    value (values) in the column(s) from cols. Case insensitive  replacements
    and/or regex replacements is supported.

    Unlike the behaviors of replace methods from a pd.Series, pd.DataFrame,
    or pd.Series.str, if parts of a value matches the target pattern, the
    entire value will be replaced. Not parts of the value.

    Parameters
    ----------
    cols : pd.Series or pandas.DataFrame
        Column(s) to replace its values.
    replacements: mapping, optional
        Mapping of pattern (keys) and its replacement (values).
    case: bool, default True
        replacements are case sensitive if True, or case insensitive if False.
        Will only apply to pairs that are both strings.
    regex: bool, default False
        Replace using regex if True. Keys will be escaped if False.
        Will only apply to pairs that are both strings.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Column(s) with replaced values.

    Examples
    --------
    >>> replacements = {'a':'1', 'B':'2', '(c)': '3'}
    >>> letters = pd.Series(['a', 'b', '(C)'], name='letters')
    >>> letters.head()
    0      a
    1      b
    2    (C)
    Name: letters, dtype: object

    Case sensitive and no regex replacements

    >>> replace_series(letters, replacements=replacements)
    0      1
    1      b
    2    (C)
    Name: letters, dtype: object

   Case insensitive and no regex replacements

    >>> replacements = {
    ...     'golden retriever': 'goldie boy',
    ...     'dove': 'lovey dovey',
    ...     '.*pug.*': 'dog that cannot breath'
    ... }
    >>> animals = pd.DataFrame({
    ...     'dogs': ['poodle', '(pug)', 'golden retriever'],
    ...     'birds': ['Pigeon', 'Crow', 'Dove']
    ... })
    >>> animals.head()
                   dogs   birds
    0            poodle  Pigeon
    1             (pug)    Crow
    2  golden retriever    Dove
    >>> replace_df(animals, replacements=replacements, case=False)
             dogs        birds
    0      poodle       Pigeon
    1       (pug)         Crow
    2  goldie boy  lovey dovey

    Case insensitive and regex replacements
    >>> replace_df(animals, replacements=replacements, case=False, regex=True)
                         dogs        birds
    0                  poodle       Pigeon
    1  dog that cannot breath         Crow
    2              goldie boy  lovey dovey
    """
    if replacements:
        try:
            return replace_df(
                cols,
                replacements=replacements,
                case=case,
                regex=regex
            )
        except AttributeError:
            return replace_series(
                cols,
                replacements=replacements,
                case=case,
                regex=regex
            )
    return cols
