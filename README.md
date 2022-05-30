# Standardize DataFrame

**standardize_df** provides features to store and execute your operations for standardizing a pandas dataframes.

## Installation

```bash
pip install standardize-df
```

## Overview

You can think of the process of ***standardizing*** a dataframe as conforming its values to some standard for comparitive evaluations. `standardize` contains many features to help standardize a dataframe:

- `adjust_df`: Functions for altering a dataframe based off an altered subset of the dataframe.
- `df_operations`: General functions to help with standardizing a dataframe.
- `pipeline`: Pipeline classes for chaining the output of one callable to the next.
- `standards`: Provides the Standard class to store and execute your operations for standardizing a dataframes column(s).
- `utils`: Utility functions related for evaluating empty values and flattening iterables

Here's a tour of some of the main features with examples from these modules:

### Adjusting DataFrames

When standardizing a dataframe, you'll often create, drop, or override columns and rows from a subset. The `adjust_df` module offers the `adjust_df` function that can reflect those changes from the subset to the dataframe.

Adjusting rows:

```python
import pandas as pd
from standardize_df.adjust_df import adjust_df
df = pd.DataFrame({'letters': ['a', 'b', 'c'], 'symbols': ['!', '@', '#']})
altered_subset = pd.Series(['!', '@'], name='symbols')
adjust_df(df, altered_subset, fields=['symbols'])
```
```
  letters symbols
0       a       !
1       b       @
```

Adjusting columns:
```python
altered_subset = pd.DataFrame({'symbols': ['!', '@', '#'], 'numbers': [1, 2, 3]})
adjust_df(df, altered_subset, fields=['symbols'])
```
```
  letters symbols  numbers
0       a       !        1
1       b       @        2
2       c       #        3
```

### Pipeline

You can store operations (callables) into `Pipeline` or `PipelineMapping` to create a data pipeline. The `PipelineMapping` offers the reorder method to easily reorder, replace, or drop operations.

Creating a data pipeline with the `Pipeline` class:
```python
from standardize_df.pipeline import Pipeline

def add_one(num): return num + 1
def add_two(num): return num + 2

pipe = Pipeline([add_one, add_two])
pipe(7)
```

    10

Creating a data pipeline with the `PipelineMapping` class:
```python
from standardize_df.pipeline import PipelineMapping

def add_one(num): return num + 1
def square_two(num): return num ** 2

pipe = PipelineMapping({'add_one': add_one, 'square_two': square_two})
pipe(2)
```
    9

Reordering a `PipelineMapping` instance with an iterable:
```python
reordered = pipe.reorder(['square_two', 'add_one'])
reordered
reordered(2)
```

    PipelineMapping([('square_two', <function square_two at 0x7f002b86e820>), ('add_one', <function add_one at 0x7f0044be15e0>)])
    5

Reordering a `PipelineMapping` instance with a mapping:

```python
reordered = pipe.reorder({'square_two': None})  # None denotes 'leave func as is'
reordered
reordered(2)
```

    PipelineMapping([('square_two', <function square_two at 0x7f002b86e820>)])
    4

### Standards
 The `standards` module offers the
`Standards` class for storing field name(s) and an operation to standardize those fields of a dataframe. The `Standards.standardize_df` method passes in a subset with the field name(s) to the operation, and adjusts the dataframe according to the return value, the altered subset.

Single field name (keys) will result in a **series** subset being passed into the operation. A series or a dataframe can be returned, and the original dataframe will be adjusted to it. 
```python
import pandas as pd
from standardize_df.standards import Standards

def drop_first(col: pd.Series) -> pd.Series: 
    '''drops the first row from the original dataframe.'''
    return col.drop(index=0)

def add_one(col: pd.Series) -> pd.DataFrame:
    '''adds the plus_one column to the original dataframe.'''
    df = col.to_frame()
    df['plus_one'] = df['numbers'] + 1
    return df

df = pd.DataFrame({'letters': ['a', 'b', 'c'], 'numbers': [1, 2, 3]})
standards_mapping = {
    'letters': drop_first, 
    'numbers': add_one
}
standards = Standards(standards_mapping)
standards.standardize_df(df)
```
      letters  numbers  plus_one
    1       b        2         3
    2       c        3         4

Multiple field name (keys) will result in a **dataframe** subset being passed into the operation. A series or dataframe can be returned, and the original dataframe will be adjusted to it. 
```python
def increment_one(df: pd.DataFrame) -> pd.DataFrame:
    '''adds columns letters_plus and numbers_plus to the original dataframe.'''
    df['letters_plus'] = df['letters'].apply(lambda x: chr(ord(x) + 1))
    df['numbers_plus'] = df['numbers'] + 1
    return df

def drop_numbers(df: pd.DataFrame) -> pd.Series:
    '''drops the numbers column from the original dataframe.'''
    return df['numbers_plus']


standards_mapping = {
    ('letters', 'numbers'): increment_one, 
    ('numbers', 'numbers_plus'): drop_numbers  # numbers_incr column added in increment_one func 
}
standards = Standards(standards_mapping)
standards.standardize_df(df)
```
      letters letters_plus  numbers_plus
    0       a            b             2
    1       b            c             3
    2       c            d             4
