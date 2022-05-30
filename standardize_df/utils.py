from typing import Any, Iterable, Iterator, Tuple, Union
import pandas as pd


def isempty(value: Any):
    """
    Returns true if value is an empty value, false otherwise. A value is
    considered empty if shares at least one of the following features:
        -> evaluates False in a boolean context; falsy.
        -> is a string with only whitespaces.
        -> null values (e.g. numpy.nan).

    Parameters
    ----------
    value : any
        value to be evaluated as empty or not

    Returns
    -------
    bool
        true if value is an empty value, false otherwise

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> isempty(np.nan)
    True
    >>> isempty(pd.NaT)
    True
    >>> isempty(pd.NA)
    True
    >>> isempty(None)
    True
    >>> isempty([])
    True
    >>> isempty('')
    True
    >>> isempty(0)
    True
    >>> isempty('nope')
    False
    """
    try:
        if not value:
            return True
        return value.isspace()
    # TypeError for pd.NA ambiguous bool value
    except (AttributeError, TypeError):
        try:
            is_na = pd.isna(value)
            if is_na is True:
                return True
            else:  # otherwise array or False
                if value:
                    return False
                return True
        except ValueError:  # catch iterable with ambiguous truth values
            return False


def has_empty(values: Iterable) -> bool:
    """
    Returns True if any item in values is empty. A value is considered empty
    if it shares at least one of the following features:
        -> evaluates False in a boolean context; falsy.
        -> is a string with only whitespaces.
        -> null values (e.g. numpy.nan).
    """
    return any([isempty(i) for i in values])


def all_empty(values: Iterable) -> bool:
    """
    Returns True if all items in values is empty. A value is considered empty
    if it shares at least one of the following features:
        -> evaluates False in a boolean context; falsy.
        -> is a string with only whitespaces.
        -> null values (e.g. numpy.nan).
    """
    return all([isempty(i) for i in values])


def remove_dup(iterable: Iterable) -> list:
    """Removes duplicates from an iterable while maintaining order."""
    return list(dict.fromkeys(iterable))


def recursive_flatten(
        iterable: Iterable,
        ignore_types: Union[type, Tuple[type, ...]] = None
) -> Iterator[Any]:
    """
    Recursively and lazily flattens, remove dimension(s), from a iterable.
    Types from ignore_types will be yielded as is.

    Parameters
    ----------
    iterable : iterable
    ignore_types : type or tuple of type, optional
        Any item with this type(s) will be yielded as is.

    Yield
    -----
    any
        an item that can't be iterated over or is of type(s) ignore_types.

    Examples
    --------
    >>> list(recursive_flatten([123, 'abc', [345, 'def']]))
    [123, 'a', 'b', 'c', 345, 'd', 'e', 'f']
    >>> list(recursive_flatten([123, 'abc', [345, 'def']], ignore_types=str))
    [123, 'abc', 345, 'def']
    """
    for i in iterable:
        if ignore_types and isinstance(i, ignore_types):
            yield i
        else:
            try:
                if len(i) == 1:
                    yield i[0]
                else:
                    yield from recursive_flatten(i, ignore_types)
            except TypeError:
                yield i


def flatten_iterable(
        iterable: Iterable,
        ignore_types: Union[None, type, Tuple[type, ...]] = (str, bytes)
) -> list:
    """
    Flattens, remove dimension(s), from a iterable. Types from ignore_types
    will be left as is.

    Parameters
    ----------
    iterable : iterable
    ignore_types : type or tuple of type, default (str, bytes).
        Optional. Any item with this type(s) will be left as is.

    Returns
    -------
    list
        Flattened list with items that you can't iterate over or is of type(s)
        ignore_types.

    Examples
    --------
    >>> flatten_iterable([123, 'abc', [345, 'def']])
    [123, 'abc', 345, 'def']
    >>> flatten_iterable([123, 'abc', [345, 'def']], ignore_types=(str, list))
    [123, 'abc', [345, 'def']]
    >>> flatten_iterable([123, 'abc', [345, 'def']], ignore_types=None)
    [123, 'a', 'b', 'c', 345, 'd', 'e', 'f']
    """
    return list(recursive_flatten(iterable, ignore_types))
