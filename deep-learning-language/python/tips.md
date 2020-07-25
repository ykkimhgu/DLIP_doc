# Tips

### List vs tuple vs dictionary in Python

**List**

* A list is a mutable, ordered sequence of items.
*  List variables are declared by using brackets `[ ]` 
* The list elements can be anything and each list element can have a completely different type. This is not allowed in arrays. Arrays are objects with definite type and size

```python
A = [ ] # This is a blank list variable
B = [2, 4, 'john'] # lists can contain different variable types.
```

**Tuple**

* Similar to list, but  _immutable_ like strings i.e. you cannot modify tuples
* Tuples are heterogeneous data structures \(i.e., their entries have different meanings\), while lists are homogeneous sequences.
* can be used as the _key_ in Dictionary
* tuple is declared in parentheses **\( \)**

```python
tupleA = (1, 2, 3, 4)
person=(‘ABC’,’admin’,’12345')

# This gives error:  'tuple' cannot be assigned
tupleA[2] = 5
```

**Dictionary**

*  A dictionary is a **key:value** pair, like an address-book. i.e. we associate _keys_ \(name\) with _values_ \(details\). 
* the key must be unique and  immutable  \(tuples, not list\)

```python
# Python 3
my_dict = {1: 'one', 2: 'two', 3: 'three'}
my_dict.keys() 
# dict_keys([1, 2, 3])
my_dict.values()
# dict_values(['one', 'two', 'three'])
```

### 

### Documenting your code in Python \(docstring\)

An official specification on how you should format your docstrings called [PEP 0257](https://www.python.org/dev/peps/pep-0257/), based on reStructuredText \(reST\)

> For Google style guide for C++, [read here](https://google.github.io/styleguide/cppguide.html)

#### Google format

```text
"""
Google Style

Args:
    param1: This is the first param.
    param2: This is a second param.

Returns:
    This is a description of what is returned.

Raises:
    KeyError: Raises an exception.
"""
```

* **Names to Avoid**

  *  dashes \(`-`\) in any package/module name
  * Use CapWords for class names, but lower\_with\_under.py for module names.

* Guidelines for naming
* | Type | Public | Internal |
  | :--- | :--- | :--- |
  | Packages | `lower_with_under` |  |
  | Modules | `lower_with_under` | `_lower_with_under` |
  | Classes | `CapWords` | `_CapWords` |
  | Exceptions | `CapWords` |  |
  | Functions | `lower_with_under()` | `_lower_with_under()` |
  | Global/Class Constants | `CAPS_WITH_UNDER` | `_CAPS_WITH_UNDER` |
  | Global/Class Variables | `lower_with_under` | `_lower_with_under` |
  | Instance Variables | `lower_with_under` | `_lower_with_under` \(protected\) |
  | Method Names | `lower_with_under()` | `_lower_with_under()` \(protected\) |
  | Function/Method Parameters | `lower_with_under` |  |
  | Local Variables | `lower_with_under` |  |
* **Example Google style docstrings**

> This module demonstrates documentation as specified by the `Google Python Style Guide`

```python
"""A one line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   https://google.github.io/styleguide/pyguide.html

"""

def function_with_types_in_docstring(param1, param2):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    .. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/

    """



def module_level_function(param1, param2=None, *args, **kwargs):
    """This is an example of a module level function.

    Function parameters should be documented in the ``Args`` section. The name
    of each parameter is required. The type and description of each parameter
    is optional, but should be included if not obvious.

    If ``*args`` or ``**kwargs`` are accepted,
    they should be listed as ``*args`` and ``**kwargs``.

    The format for a parameter is::

        name (type): description
            The description may span multiple lines. Following
            lines should be indented. The "(type)" is optional.

            Multiple paragraphs are supported in parameter
            descriptions.

    Args:
        param1 (int): The first parameter.
        param2 (:obj:`str`, optional): The second parameter. Defaults to None.
            Second line of description should be indented.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        bool: True if successful, False otherwise.
        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::

            {
                'param1': param1,
                'param2': param2
            }

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions
            that are relevant to the interface.
        ValueError: If `param2` is equal to `param1`.

    """
    if param1 == param2:
        raise ValueError('param1 may not be equal to param2')
    return True


```



