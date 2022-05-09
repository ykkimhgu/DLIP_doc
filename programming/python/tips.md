# Tips

## List vs tuple vs dictionary in Python

**List**

* A list is a mutable, ordered sequence of items.
* List variables are declared by using brackets `[ ]`
* The list elements can be anything and each list element can have a completely different type. This is not allowed in arrays. Arrays are objects with definite type and size

```python
A = [ ] # This is a blank list variable
B = [2, 4, 'john'] # lists can contain different variable types.
```

**Tuple**

* Similar to list, but _immutable_ like strings i.e. you cannot modify tuples
* Tuples are heterogeneous data structures (i.e., their entries have different meanings), while lists are homogeneous sequences.
* can be used as the _key_ in Dictionary
* tuple is declared in parentheses **( )**

```python
tupleA = (1, 2, 3, 4)
person=(‘ABC’,’admin’,’12345')

# This gives error:  'tuple' cannot be assigned
tupleA[2] = 5
```

**Dictionary**

* A dictionary is a **key:value** pair, like an address-book. i.e. we associate _keys_ (name) with _values_ (details).
* written with curly brackets { }
* A colon (:) separates each **key** from its **value**.
* the key must be unique and immutable (tuples, not list)

```python
# Python 3
my_dict = {1: 'one', 2: 'two', 3: 'three'}
my_dict.keys() 
# dict_keys([1, 2, 3])
my_dict.values()
# dict_values(['one', 'two', 'three'])
```

## Dictionary in Python

### Create a dictionary with dict() constructor

```python
# create a dictionary using two list
students = ['Amanda', 'Teresa', 'Paula', 'Mario']
ages = [27, 38, 17, 40]

# zip method --> iterator of tuples --> dict method --> dictionary
students_ages = dict(zip(students, ages))
print(students_ages)
# {'Amanda': 27, 'Teresa': 38, 'Paula': 17, 'Mario': 40}


# create a dictionary with dict() function using keyword arguments
# dictionary - ages of students
students_ages = dict(Amanda=27, Teresa=38, Paula=17, Mario=40)
print(students_ages)
# {'Amanda': 27, 'Teresa': 38, 'Paula': 17, 'Mario': 40}


 students_ages = {'Amanda': 27, 'Teresa': 38, 'Paula': 17, 'Mario': 40}
 print(students_ages)
```

## Class in Python

### **Instance methods**

* Called using objects
* Must have `self` as the first parameter
* (`self` is another python term. We can use self to access any data or other instance methods which resides in that class. These cannot be accessed without using self)

### Initializer Method

* must be called `__init__()`\~\~(\~\~double underscore is used by python runtime)
* The first parameter is `self`
* If the initializer method is present, the constructor calls `__init__()`

### Super() to inherit all the methods and properties from another class:

* Inherits all the method, properties of Parent or sibling class

```python
class Child(Parent):
  def __init__(self, txt):
    super().__init__(txt)  # inherit Parent;s method/properties
```

`>` _\*\*\_The underscore prefix in a variable/method name is meant as a \_hint_ to another programmer that a variable or method starting with a single underscore is intended only for internal use. This convention is [defined in PEP 8](http://pep8.org/#descriptive-naming-styles).

```python
# Base class
class House:
    '''
    A place which provides with shelter or accommodation
    '''
    def __init__(self, rooms, bathrooms):
        self.rooms = rooms
        self.bathrooms = bathrooms
    def room_details(self):
        print(f'This property has {self.rooms} rooms \
              with {self.bathrooms} bathrooms')
class Apartment(House):
    '''
    A house within a large building where others also have
    their own house
    '''
    def __init__(self, rooms, bathrooms, floor):
        House.__init__(self, rooms, bathrooms)
        self.floor = floor

# Create an Apartment
apartment = Apartment(2, 2, 21)
apartment.room_details()

Output:
This property has 2 rooms with 2 bathrooms
```

## Index of array

### .shape

input\_tensor.shape\[-1] # get value of the last index of shape

## Reshaping arrays

source:[ read here](https://towardsdatascience.com/reshaping-numpy-arrays-in-python-a-step-by-step-pictorial-tutorial-aed5f471cf0b)

![source click here](<../../images/image (9).png>)

## Stacking 2D data to 3D data

```python
 # change to  [rows][cols][channels] for Keras

    # Method0
    x_train3D=np.stack((x_train,x_train,x_train),axis=2) 

    # Method1
    # numpy(channel,r,c) [channels].[rows][cols]
    x_train3D=np.stack((x_train,x_train,x_train))  
    print(x_train3D.shape)    
    x_train3D=np.moveaxis(x_train3D,0,2)
    print(x_train3D.shape)

    x_test3D=np.stack((x_test,x_test,x_test))
    x_test3D=np.moveaxis(x_test3D,0,2)
    print(x_test3D.shape)


    # NEEDS TO BE MODIFIED  (stack-->concatenate)
    # Method2
    x_train=np.expand_dims(x_train,axis=2)
    print(x_train.shape)
    x_train3D=np.stack((x_train,x_train,x_train),axis=1)  
    print(x_train3D.shape)

    # Method3
    x_train=np.reshape(x_train,[x_train.shape[0],x_train.shape[1],1])
    print(x_train.shape)
    x_train3D=np.stack((x_train,x_train,x_train))  
    print(x_train3D.shape)
```

## Documenting your code in Python (docstring)

An official specification on how you should format your docstrings called [PEP 0257](https://www.python.org/dev/peps/pep-0257/), based on reStructuredText (reST)

> For Google style guide for C++, [read here](https://google.github.io/styleguide/cppguide.html)

### Google format

```
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
  * dashes (`-`) in any package/module name
  * Use CapWords for class names, but lower\_with\_under.py for module names.
* Guidelines for naming
* | Type                       | Public               | Internal                          |
  | -------------------------- | -------------------- | --------------------------------- |
  | Packages                   | `lower_with_under`   |                                   |
  | Modules                    | `lower_with_under`   | `_lower_with_under`               |
  | Classes                    | `CapWords`           | `_CapWords`                       |
  | Exceptions                 | `CapWords`           |                                   |
  | Functions                  | `lower_with_under()` | `_lower_with_under()`             |
  | Global/Class Constants     | `CAPS_WITH_UNDER`    | `_CAPS_WITH_UNDER`                |
  | Global/Class Variables     | `lower_with_under`   | `_lower_with_under`               |
  | Instance Variables         | `lower_with_under`   | `_lower_with_under` (protected)   |
  | Method Names               | `lower_with_under()` | `_lower_with_under()` (protected) |
  | Function/Method Parameters | `lower_with_under`   |                                   |
  | Local Variables            | `lower_with_under`   |                                   |
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
