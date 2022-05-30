from __future__ import annotations
from collections import MutableSequence, OrderedDict
from functools import partial
import reprlib
from typing import \
    Any, \
    Callable, \
    Hashable, \
    Iterable, \
    Mapping, \
    MutableMapping, \
    Sequence, \
    Union, \
    Iterator


class Pipeline(MutableSequence):
    """
    A List-like container of operations (callables) that can be passed in only
    one argument.

    Behaves like a callable, taking the output of an operation and passing
    it into the next operation in line.

    Class Attributes
    ----------------
    enforce_types : type or tuple of type, default None
        If provided, validates obj and return values from each operation is of
        these types.
    """
    enforce_types = None

    def __init__(self, pipeline: Iterable[Callable]) -> None:
        self._pipeline = []
        for n, i in enumerate(list(pipeline)):
            self.insert(n, i)

    def insert(self, index: int, o: Callable) -> None:
        if callable(o):
            self._pipeline.insert(index, o)
        else:
            raise TypeError(f'Expected a callable, got {type(o)}')

    def __getitem__(self, i: Union[int, slice]) -> Callable:
        if isinstance(i, slice):
            cls = type(self)
            return cls(self._pipeline[i])
        return self._pipeline[i]

    def __setitem__(self, i: int, o: Callable) -> None:
        if callable(o):
            self._pipeline[i] = o
        else:
            raise TypeError(f'Expected a callable, got {type(o)}')

    def __delitem__(self, i: int) -> None:
        del self._pipeline[i]

    def __len__(self) -> int:
        return len(self._pipeline)

    def __add__(self, other: Pipeline) -> Pipeline:
        if isinstance(other, Pipeline):
            cls = type(self)
            return cls(self._pipeline + list(other))
        return NotImplemented

    def __radd__(self, other: Pipeline) -> Pipeline:
        return self + other

    def __iadd__(self, other: Iterable) -> Pipeline:
        if isinstance(other, Pipeline):
            other = other[:]  # prevent infinite loop if other is self.
        else:
            try:
                other = iter(other)
            except TypeError:
                raise TypeError('right operand in += must be an iterable')
        for n, i in enumerate(other, start=len(self) - 1):
            self.insert(n, i)
        return self

    def __mul__(self, other: int) -> Pipeline:
        if isinstance(other, int):
            cls = type(self)
            return cls(self._pipeline * other)
        else:
            type_name = type(other).__name__
            msg = f"can't multiply sequence by non-int of type '{type_name}'"
            raise TypeError(msg)

    def __rmul__(self, other: int) -> Pipeline:
        return self * other

    def __imul__(self, other: int) -> Pipeline:
        if isinstance(other, int):
            self._pipeline *= other
        else:
            type_name = type(other).__name__
            msg = f"can't multiply sequence by non-int of type '{type_name}'"
            raise TypeError(msg)
        return self

    def __eq__(self, other: Iterable) -> bool:
        return len(self) == len(other) and \
               all(a == b for a, b in zip(self, other))

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        return '{}({})'.format(
            cls_name,
            self._pipeline
        )

    def __str__(self) -> str:
        cls_name = type(self).__name__
        return '{}({})'.format(
            cls_name,
            reprlib.repr(self._pipeline)
        )

    def __call__(self, obj: Any) -> Any:
        """
        Connects a series of external operations and passes in the result of
        each operation to the next.

        Arguments
        ---------
        obj : Any
            An object to pass into the first operation of the pipeline.

        Returns
        -------
        Any
            Should be the type of obj.

        Raises
        ------
        TypeError
            If result type from an operation does not match self.enforce_type
            and self.enforce_type has a value.

        Examples
        --------
        >>> add3 = Pipeline([lambda x: x + 1, lambda x: x + 2])
        >>> add3(1)
        4
        >>> def increment_letter(char, n):
        ...     return chr(ord(char) + n)
        ...
        >>> operations = [
        ...     lambda char: increment_letter(char, n=2),
        ...     lambda char: increment_letter(char, n=3)
        ... ]
        >>> increment_five = Pipeline(operations)
        >>> increment_five('a')
        'f'
        >>> class StringPipelineSequence(Pipeline):
        ...     enforce_types = str
        ...
        >>> make_error = StringPipelineSequence([lambda x: x, lambda x: 3])
        >>> make_error('apple')  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        TypeError: Result from operations '<function <lambda> at ...>' does not
        match any of the enforced types '<class 'str'>', got: <class 'int'>
        """
        if self.enforce_types and not isinstance(obj, self.enforce_types):
            msg = f"obj does not match any of the enforced types " \
                  f"'{self.enforce_types}', got: {type(obj)}"
            raise TypeError(msg)

        for operation in self._pipeline:
            obj = operation(obj)
            if self.enforce_types and not isinstance(obj, self.enforce_types):
                msg = f"Result from operations '{operation}' does not match " \
                      f"any of the enforced types '{self.enforce_types}', " \
                      f"got: {type(obj)}"
                raise TypeError(msg)
        return obj


class PipelineMapping(MutableMapping):
    """
    An OrderedDict-like of operation names (keys) and their operations (values).

    Behaves like a callable, taking the output of an operation and passing
    it into the next operation in line.

    Class Attributes
    ----------------
    enforce_types : type or tuple of type, default None
        If provided, validates obj and return values from each operation is of
        these types.

    Methods
    -------
    reorder(self, order) -> PipelineMapping:
        Re-orders the operation order and returns a new PipelineMapping instance.
    set_partial(self, k, v, *args, **kwargs):
        Create a partial function out of v and sets it to k.
    """

    enforce_types = None

    def __init__(
            self,
            pipeline: Mapping[Hashable, Callable] = None,
            **kwargs
    ) -> None:
        """
        Constructor for the pipeline.

        pipeline : Mapping
            A mapping of operation names (keys) and their operations (values).
        **kwargs
            Keyword arguments of operation names (keys) and their
            operations (values).
        """
        if pipeline:
            self._pipeline = OrderedDict(pipeline)
        else:
            self._pipeline = OrderedDict()
        if kwargs:
            self._pipeline.update(**kwargs)

    def __len__(self) -> int:
        return len(self._pipeline)

    def __getitem__(self, k: Hashable) -> Callable:
        return self._pipeline[k]

    def __setitem__(self, k: Hashable, v: Callable) -> None:
        self._pipeline[k] = v

    def __delitem__(self, k: Hashable) -> None:
        del self._pipeline[k]

    def __iter__(self) -> Iterator:
        return iter(self._pipeline)

    def __call__(self, obj: Any) -> Any:
        """
        Connects a series of external operations and passes in the result of
        each operation to the next.

        Arguments
        ---------
        obj : Any
            An object to pass into the first operation of the pipeline.

        Returns
        -------
        Any
            Should be the type of obj.

        Raises
        ------
        TypeError
            If result type from an operation does not match self.enforce_type
            and self.enforce_type has a value.

        Examples
        --------
        >>> add3 = PipelineMapping(add_one=lambda x: x + 1, add_two=lambda x: x + 2)
        >>> add3(1)
        4
        >>> def increment_letter(char, n):
        ...     return chr(ord(char) + n)
        ...
        >>> increment_five = PipelineMapping(
        ...     increment_two=lambda char: increment_letter(char, n=2),
        ...     increment_three=lambda char: increment_letter(char, n=3)
        ... )
        >>> increment_five('a')
        'f'
        >>> class StringPipeline(PipelineMapping):
        ...     enforce_types = str
        ...
        >>> make_error = StringPipeline(
        ...     return_str=lambda x: x,
        ...     return_int=lambda x: 3
        ... )
        >>> make_error('apple')  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        TypeError: Result from operations 'return_int' does not match any of
        the enforced types '<class 'str'>', got: <class 'int'>
        """
        if self.enforce_types and not isinstance(obj, self.enforce_types):
            msg = f"obj does not match any of the enforced types " \
                  f"'{self.enforce_types}', got: {type(obj)}"
            raise TypeError(msg)

        for op_name, operation in self.items():
            obj = operation(obj)
            if self.enforce_types and not isinstance(obj, self.enforce_types):
                msg = f"Result from operations '{op_name}' does not match " \
                      f"any of the enforced types '{self.enforce_types}', " \
                      f"got: {type(obj)}"
                raise TypeError(msg)
        return obj

    def __repr__(self) -> str:
        return "{cls.__name__}({components})".format(
            cls=type(self),
            components=list(self._pipeline.items())
        )

    def move_to_end(self, key: Hashable, last: bool = True) -> None:
        """
        Move an existing element to the end (or beginning if last is false).
        Raise KeyError if the element does not exist.
        """
        self._pipeline.move_to_end(key, last)

    def reorder(self, order: Union[MutableMapping, Sequence]) -> PipelineMapping:
        """
        Rearranges the pipeline into the given order.

        Parameters
        ----------
        order : MutableMapping or Sequence
            A pipeline to override, re-order, or exclude operations only found
            in the pipeline. A sequence can re-order and exclude operations.
            Mappings can override, re-order, or exclude  operations. To keep
            your mapping from overriding and excluding an operation, set the
            operation function to None.

        Returns
        -------
        PipelineMapping
            Rearranged pipeline.

        Raises
        ------
        KeyError
            If any operation name does not already exist within the pipeline.

        Examples
        --------
        Create a new pipeline

        >>> def square_minus_one(n):
        ...     return n ** 2 - 1
        ...
        >>> def cube_minus_one(n):
        ...     return n ** 3 - 1
        ...
        >>> calculate = PipelineMapping(
        ...     calc1=square_minus_one,
        ...     calc2=cube_minus_one
        ... )
        >>> calculate  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        PipelineMapping([('calc1', <function square_minus_one at ...>),
        ('calc2', <function cube_minus_one at ...>)])
        >>> calculate(2)
        26

        Reorder with a mapping (no overwrite)

        >>> calculate = calculate.reorder({'calc2': None, 'calc1': None})
        >>> calculate  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        PipelineMapping([('calc2', <function cube_minus_one at ...>),
        ('calc1', <function square_minus_one at ...>)])
        >>> calculate(2)
        48

        Reorder with a sequence

        >>> calculate = calculate.reorder(['calc1', 'calc2'])
        >>> calculate  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        PipelineMapping([('calc1', <function square_minus_one at ...>),
        ('calc2', <function cube_minus_one at ...>)])
        >>> calculate(2)
        26

        Reorder and overwrite with a mapping

        >>> def biquadrate_minus_one(n):
        ...     return n ** 4 - 1
        ...
        >>> order = {'calc2': None, 'calc1': biquadrate_minus_one}
        >>> calculate = calculate.reorder(order)
        >>> calculate  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        PipelineMapping([('calc2', <function cube_minus_one at ...>),
        ('calc1', <function biquadrate_minus_one at ...>)])
        >>> calculate(2)
        2400

        Providing a new operation name raises KeyError

        >>> order = {'calc3000': square_minus_one}
        >>> calculate.reorder(order) # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        KeyError: 'Cannot override keyword(s) calc3000, it does not exist in
        the pipeline.'
        """
        cls = type(self)
        return cls(reorder(self, order))

    def set_partial(self, k: Hashable, v: Callable, *args, **kwargs) -> None:
        """
        Create a partial function out of v and sets it to k.

        Arguments
        ---------
        k : Hashable
        v : Callable
            Callable to freeze arguments to.
        *args
            Positional arguments to freeze on v.
        **kwargs
            Keyword arguments to freeze on v.
        """
        self.__setitem__(k, partial(v, *args, **kwargs))


def reorder(
        d_: Mapping,
        order: Union[MutableMapping, Sequence]
) -> OrderedDict:
    """
    Rearranges a mapping into the given order.

    Parameters
    ----------
    d_ : MutableMapping
    order : MutableMapping or Sequence
        A pipeline to override, re-order, or exclude operations only found in
        the pipeline. A sequence can re-order and exclude operations. Mappings
        can override, re-order, or exclude  operations. To keep your mapping
        from overriding and excluding an operation, set the operation function
        to None.

    Returns
    -------
    OrderedDict
        Rearranged OrderedDict.

    Raises
    ------
    KeyError
        If any operation name does not already exist within the mapping,
        raise this error.

    Examples
    --------
    >>> d = OrderedDict({'numbers': [1, 2, 3], 'letters': ['a', 'b', 'c']})
    >>> # Reorder with a mapping (no overwrite)
    >>> reorder(d, order={'letters': None, 'numbers': None})
    OrderedDict([('letters', ['a', 'b', 'c']), ('numbers', [1, 2, 3])])
    >>> # Reorder with a sequence
    >>> reorder(d, order=['letters', 'numbers'])
    OrderedDict([('letters', ['a', 'b', 'c']), ('numbers', [1, 2, 3])])
    >>> # Reorder and overwrite with a mapping
    >>> reorder(d, order={'letters': ['x', 'y', 'z'], 'numbers': None})
    OrderedDict([('letters', ['x', 'y', 'z']), ('numbers', [1, 2, 3])])
    >>> # Providing a new operation name raises KeyError
    >>> reorder(d, order={'signs': ['!', '@', '#']})  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    KeyError: 'Cannot override keyword(s) signs, it does not exist in ...'
    """

    new = OrderedDict()
    try:
        for k, func in order.items():
            if k in d_.keys():
                if func:
                    new[k] = func
                else:
                    new[k] = d_[k]
            else:
                msg = f"Cannot override keyword(s) {k}, " \
                      f"it does not exist in the pipeline."
                raise KeyError(msg)
        return new
    except AttributeError:
        for k in order:
            if k in d_.keys():
                new[k] = d_[k]
            else:
                msg = f"Cannot override keyword(s) {k}, " \
                      f"it does not exist in the pipeline."
                raise KeyError(msg)
        return new
