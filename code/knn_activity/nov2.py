# Structure and Intepretation of Computer Programs

'''
free: not bound with actual value
bound: the variable has beem assigned
pipeline

'''

def add2(x):
    return x + 2


def half(x):
    return x / 2

def compose(inner, outer):
    return lambda x: outer(inner(x))

# def compose_1(f1, f2, f3):
#     f = compose(f1, f2)
#     return compose(f, f3)
    
# def comp(f1, f2, f3):
#     f = compose(f1, f2)
#     return lambda x: f3(f(x))

def add_maker(n):
    # x is free, n is bound by whatever value being passed in as a param.
    return lambda x: x + n

def div_maker(n):
    return lambda x: x / n

def iterate(f, n):
    out_f = f
    for i in range(n - 1):
        out_f = compose(f, out_f)
    return out_f

'''
0: half(add2(x))
1: quarter(half(add2(x)))
'''
def compose_func_list(flist):
    f = flist[0]
    for i in range(1, len(flist)):
        f = compose(f, flist[i]) 
    return f

def reduce_func_list(f, values):
    output = f(values[0], values[1])
    for i in range(2 , len(values)):
        output = f(output, values[i])
    return output

def main():
    add2 = add_maker(2)
    half = div_maker(2)
    quarter = div_maker(4)
    f_compose = compose(add2, half)
    print(f_compose(5))
    f_list = compose_func_list([add2, half, quarter])
    print(f_list(5))
    f_n = iterate(add2, 3)
    print(f_n(5))
    
main()