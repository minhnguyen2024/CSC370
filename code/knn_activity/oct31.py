import pandas as pd
import numpy as np

def add2(a):
    return a + 2

def compose(f1,f2,x):
    return f2(f1(x))

def composer(f1, f2):
    # def new_f(x):
    #     return f2(f1(x))
    # return new_f
    return lambda x: f2(f1(x))

def main():
    result = compose(lambda x: x + 2, lambda x: x**2, 3)

    print(result)


main()
