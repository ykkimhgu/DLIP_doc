# Python Tutorial - For Loop

## Python Tutorial - For loop

### Reference

점프 투 파이썬:[박응용](https://wikidocs.net/profile/info/book/3)

https://wikidocs.net/book/1

## Overview

* iterable은 사전적의미와 똑같이 반복가능한 객체를 말합니다.
* list, dictionary, set, string, tuple가 iterable한 타입입니다.

```python
for item in iterable:
  ... 반복할 구문...
```

* **range**도 iterable 합니다.
* range는 `range(시작숫자, 종료숫자, step)`의 형태로 리스트 슬라이싱과 유사합니다.
* range의 결과는 시작숫자부터 종료숫자 바로 앞 숫자까지 컬렉션을 만듭니다.

```python
for i in range(5):
...     print(i)
```

* **enumerate** : 몇 번째 반복문인지 확인 하기 위해 인덱스 번호와 컬렉션의 원소를 tuple형태로 반환합니다.

```python
for p in enumerate(t):
...     print(p)

```

### Examples

```python
>>> for i in range(3):
...     print(i)
... 
0
1
2

>>> t = [1, 5, 7]
>>> for p in enumerate(t):
...     print(p)
... 
(0, 1)
(1, 5)
(2, 7)

>>> for i, v in enumerate(t):
...     print("index : {}, value: {}".format(i,v))
... 
index : 0, value: 1
index : 1, value: 5
index : 2, value: 7
```
