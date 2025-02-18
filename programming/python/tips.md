# Python Tutorial - Tips

## if \_\_name\_\_=="\_main\_"



모듈이 **직접 실행**되었는지 혹은  **import** 되었는지 아닌지 판단할 때 `__name__` 변수의 값을 사용합니다.



일반적으로, 모듈은 직접 실행되거나 다른 모듈에서 import 되어 사용됩니다. 만약 모듈이 직접 실행되면, `__name__` 변수는 문자열`"__main__"`이 할당됩니다. 반대로, 모듈이 import 되어 사용될 때는,`__name__`변수는 해당 모듈의 이름(파일명)이 할당됩니다.

따라서, `__name__` 변수의 값을`"__main__"`과 비교하면 현재 모듈이 직접 실행되는지(import 되는지)를 판단할 수 있습니다. 따라서 코드를 if`name == "main"로`감싸면, 해당 파일이 모듈로 사용될 때는 실행되지 않고, 직접 실행될 때만 실행됩니다.

**장점:**

* 모듈로 사용될 때는 코드가 실행되지 않으므로, 다른 모듈에서 해당 모듈을 import 할 때 발생할 수 있는 부작용을 방지할 수 있습니다.
* 모듈을 개발할 때, 테스트 코드를 추가하고 싶을 때,`if name == "main":`구문을 활용하여, 해당 모듈을 직접 실행할 때만 테스트 코드가 실행되도록 할 수 있습니다.

```python
# calc.py
def add(a, b):
    return a + b 
def mul(a, b):
    return a * b

print('calc.py __name__:', __name__)    # __name__ 변수 출력 
if __name__ == '__main__':    # 프로그램의 시작점일 때만 아래 코드 실행
    print(add(10, 20))
    print(mul(10, 20))
    



# cal_main.py
import calc
print('calc_main.py __name__:', __name__)    # __name__ 변수 출력
calc.add(50, 60)
```

```python
## calc.py 실행시
# 실행결과
calc.py __name__: calc
30
200


## calc_main.py 실행시
calc_main.py __name__: calc_main
110   # >>> calc.add(50, 60)
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

## NP Array

###

### Reshaping arrays

#### .shape

input\_tensor.shape\[-1] # get value of the last index of shape

source:[ read here](https://towardsdatascience.com/reshaping-numpy-arrays-in-python-a-step-by-step-pictorial-tutorial-aed5f471cf0b)

![source click here](<../../.gitbook/assets/image (9).png>)

### Stacking 2D data to 3D data

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
