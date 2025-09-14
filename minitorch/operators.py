"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x1, x2):
    return x1 * x2


def id(x):
    return x


def add(x1, x2):
    return x1 + x2


def neg(x):
    return -x


def lt(x1, x2):
    return x1 < x2


def eq(x1, x2):
    return x1 == x2


def max(x1, x2):
    return x2 if lt(x1, x2) else x1


def is_close(x1, x2, eps=1e-2):
    return abs(x1 - x2) < eps


def exp(x):
    return math.exp(x)


def log(x):
    return math.log(x)


def inv(x):
    return 1.0 / x


def sigmoid(x):
    return (
        inv((add(1.0, exp(neg(x)))))
        if lt(0.0, x)
        else mul(exp(x), inv(add(1.0, exp(x))))
    )


def relu(x):
    return max(0.0, x)


def log_back(x1, x2):
    """
    x2 * d [log(x1)] / dx1
    """
    return mul(inv(x1), x2)


def inv_back(x1, x2):
    """
    x2 * d [1/x1] / dx1
    """
    return mul(neg(inv(mul(x1, x1))), x2)


def relu_back(x1, x2):
    """
    x2 * d ReLU(x1) dx
    """
    if lt(x1, 0.0):
        return 0.0
    return x2


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(func: Callable, x: Iterable):
    result = []
    for i in x:
        result.append(func(i))
    return result


def zipWith(func: Callable, x: Iterable, y: Iterable):
    result = []
    for i, j in zip(x, y):
        result.append(func(i, j))
    return result


def reduce(func: Callable, x: Iterable):
    it = iter(x)
    try:
        result = next(it)
    except:
        return None
    for i in it:
        result = func(result, i)
    return result


def negList(x: list):
    return map(neg, x)


def addLists(x: list, y: list):
    return zipWith(add, x, y)


def sum(x: list):
    return reduce(add, x) if len(x) > 0 else 0


def prod(x: list):
    return reduce(mul, x)
