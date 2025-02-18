# Python Tutorial - List Tuple, Dic, Set

## Python Tutorial - List, Set, Dict, Tuple

### Reference

점프 투 파이썬:[박응용](https://wikidocs.net/profile/info/book/3)

https://wikidocs.net/book/1

## Overview

Python 객체에는 mutable과 immutable 객체가 있습니다.

* Mutable: 메모리 공간에서 값 수정 가능
* Immutable: 메모리 공간에서 값 수정 불가능. 기존 변수값을 수정할 시 새로운 메모리에서 생성됨

객체 집합은, mutable 및 순서 기능에 따라 다양한 객체형태로 구분됩니다.

| class | 선언    |                 설명                 |     구분    |
| :---: | ----- | :--------------------------------: | :-------: |
|  list | \[ ]  |        mutable, 순서가 있는 객체 집합       |  mutable  |
| tuple | ( )   |       immutable, 순서가 있는 객체 집합      | immutable |
|       |       |                                    |           |
|  dict | { }   | mutable, 순서 없고, key와 value가 맵핑된 객체 |  mutable  |
|  set  | { : } |      mutable, 순서 없는 고유한 객체 집합      |  mutable  |

#### List

* 순서가 있는 수정가능한 객체의 집합입니다.
* list 는 `[]` 대괄호로 작성되어지며, 내부 원소는 `,`로 구분됩니다.

#### Tuple

* tuple(튜플)은 불변한 순서가 있는 객체의 집합입니다.
* list형과 비슷하지만 한 번 생성되면 값을 변경할 수 없습니다.

#### Dictionary

* immutable한 키(key)와 mutable한 값(value)으로 맵핑되어 있는 순서가 없는 집합입니다.
* 키로는 immutable한 값은 사용할 수 있지만, mutable한 객체는 사용할 수 없습니다.
* 중괄호로 되어 있고 키와 값이 있습니다. `{"a" : 1, "b":2}`

#### Set

* 순서가 없고, 집합안에서는 unique한 값을 가집니다.
* 그리고 mutable 객체입니다.
* 중괄호를 사용하는 것은 dictionary와 비슷하지만, key가 없습니다. 값만 존재합니다. `s = {3, 5, 7}`

### Examples

```python
# List
a = [1, 3, 5, 7]
b= ['c', 354, True]

# Tuplet
t = (1, "korea", 3.5, 1)


# Dict
dic_a = {1: 5, 2: 3}  
dic_b = {(1,5): 5, (3,3): 3}  # immutable tuple만 Key로 가능
dic_c = { 3.6: 5, "abc": 3}

# Set
s = {3, 5, 7}
s = {"1", 3, 5, (1,3)} # immutable tuple만 element로 가능
```

***

## LIST 사용예시

* 순서가 있는 수정가능한 객체의 집합입니다.
* list 는 `[]` 대괄호로 작성되어지며, 내부 원소는 `,`로 구분됩니다.

#### Index, Slicing

* `리스트변수[시작인덱스:종료인덱스:step]`
* 종료인덱스의 원소는 **포함되지 않고** 바로 앞 원소까지만 포함됩니다. step은 생략됩니다.

```python
a = [1, 3 , 5, 7]  # a[0] ~ a[3] 접근 가능
a[-1]   	# >> 7    -1 역순 인덱싱
a[1:-1] 	# >> [3,5,7]
a[:2]		# >> [1,3]
a.index(3)  # >> 1


a.append(5)  # >> a = [1, 3 , 5, 7, 5] 
a+ [ 2, 7]   # >> a=[1, 3, 5, 7, 2, 7]
list("가나다") # >> ['가', '나', '다']



```

## Tuple 사용예시

* List와 거의 동일하지만, 튜플은 요솟값을 바꿀 수 없다.
* const 성능으로 사용할때 유용함

```python
ta = (1, "korea", 3.5, 1)
tb= 1,2,5		# tuple로 인식
tc = (1,)		# 1개 element 이면, 쉼표가 있어야 함
```

## Dictionary 사용예시

* 값은 중복될 수 있지만, 키가 중복되면 마지막 값으로 덮어씌워집니다.
* 순서가 없기 때문에 인덱스로는 접근할수 없고, 키로 접근 할 수 있습니다.
* 키로는 immutable한 값은 사용할 수 있지만, mutable한 객체는 사용할 수 없습니다.

```python
dica={"a" : 1, "b":2}
dicb = { True: 5, "abc": 3} 

dicc = {[1,3]: 5, [3,5]: 3} # gives ERROR, used mutable list in KEY
dicc = {'a': [1, 2, 3]}  # OK, to use mutable list in VALUE

dicd={"a" : 1, "a":2}  # {'a': 2},  Key 중복은 덮어짐

querya= dica[0]  		# gives ERROR
querya= dica['a']		# >> querya=1

dica= {'name': 'pey', 'phone': '010-9999-1234', 'birth': '1118'}
dica.keys()				# >> dict_keys(['name', 'phone', 'birth'])
dica.values()			# >> dict_values(['pey', '010-9999-1234', '1118'])
dica.items()			# Key와 Value의 쌍을 튜플로 묶은 값을 dict_items 객체로 리턴한다. >> dict_items([ ...])

for k in dica.keys():
...    print(k) 		# >> name, phone, birth
```

## Set 사용예시

*   순서가 없고, 집합안에서는 unique한 값을 가집니다.

    > set은 중복을 허용하지 않는 특징 때문에 데이터의 중복을 제거하기 위한 필터로 종종 사용된다
* 중괄호를 사용하는 것은 dictionary와 비슷하지만, key가 없습니다. 값만 존재합니다.
* set(집합) 내부 원소는 다양한 값을 함께 가질 수 있지만, mutable한 값은 가질수 없습니다.
* 리스트나 튜플은 순서가 있기(ordered) 때문에 인덱싱을 통해 요솟값을 얻을 수 있지만, set 자료형은 순서가 없기(unordered) 때문에 인덱싱을 통해 요솟값을 얻을 수 없다.
  * 만약 set 자료형에 저장된 값을 인덱싱으로 접근하려면 다음과 같이 리스트나 튜플로 변환한 후에 해야 한다.

```python
s1 = {3, 5, 7}
s2 = set([1, 2, 3])
s3 = set("Hello") 	# >> {'e', 'H', 'l', 'o'} , 중복 허용하지 않고, 순서가 없다


l1 = list(s1)	# >> [3, 5, 7]
l1[0] 			# >> 3

s4 = set([1, 2, 3])
s4.update([4, 5, 6])  # >> s4={1, 2, 3, 4, 5, 6}
s4.add(50)			 # >>> s4={1, 50, 3}
```

## 19. for in 반복문, Range, enumerate

### 1. for in 반복문

* 여타 다른 언어에서는 일반적인 for문, foreach문, for of문등 여러가지 방식을 한꺼번에 지원하는 경우가 많습니다.
* Python에서는 for in문 한가지 방식의 for 문만 제공합니다.
* REPL 에서 확인해보겠습니다.
* for in 문 형식 입니다.
* iterable은 사전적의미와 똑같이 반복가능한 객체를 말합니다.

```python
for item in iterable:
  ... 반복할 구문...
```

* iterable 객체를 판별하기 위해서는 아래의 방법이 있습니다.
* collections.Iterable에 속한 instance인지 확인하면 됩니다.
* isinstance 함수는 첫번째 파라미터가 두번째 파라미터 클래스의 instance이면 True를 반환합니다.
* 앞서 다룬 타입 중 list, dictionary, set, string, tuple, bytes가 iterable한 타입입니다.
* range도 iterable 합니다. 이번 포스트 아래쪽에서 다루겠습니다.
* for문을 동작시켜봅니다.

```python
Copy>>> for i in var_list:
...     print(i)
... 
1
3
5
7
```

### 2. range

* 위쪽 for문의 range 결과 값이 0, 1, 2, 3, 4 순서대로 결과 값이 출력되었습니다.
* range는 `range(시작숫자, 종료숫자, step)`의 형태로 리스트 슬라이싱과 유사합니다.
* range의 결과는 시작숫자부터 종료숫자 바로 앞 숫자까지 컬렉션을 만듭니다.
* 시작숫자와 step은 생략가능합니다.

```python
>>> range(5)
range(0, 5)
>>> for i in range(5):
...     print(i)
... 
0
1
2
3
4
```

* range는 값을 확인하기 위해서 다른 순서 있는 컬렉션으로 변환해야합니다.

```python
>>> range(5,10)
range(5, 10)
>>> list(range(5,10))
[5, 6, 7, 8, 9]
>>> tuple(range(5,10))
(5, 6, 7, 8, 9)
```

* step을 사용해봅니다.

```python
Copy>>> list(range(10,20,2))
[10, 12, 14, 16, 18]
>>> list(range(10,20,3))
[10, 13, 16, 19]
```

### 3. enumerate

* 반복문 사용 시 몇 번째 반복문인지 확인이 필요할 수 있습니다. 이때 사용합니다.
* 인덱스 번호와 컬렉션의 원소를 tuple형태로 반환합니다.

```python
>>> t = [1, 5, 7, 33, 39, 52]
>>> for p in enumerate(t):
...     print(p)
... 
(0, 1)
(1, 5)
(2, 7)
(3, 33)
(4, 39)
(5, 52)
```

* tuple형태 반환을 이용하여 아래처럼 활용할 수 있습니다.

```python
>>> for i, v in enumerate(t):
...     print("index : {}, value: {}".format(i,v))
... 
index : 0, value: 1
index : 1, value: 5
index : 2, value: 7
index : 3, value: 33
index : 4, value: 39
index : 5, value: 52
```
