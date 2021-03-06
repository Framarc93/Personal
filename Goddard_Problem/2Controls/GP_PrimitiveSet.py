import numpy as np

def lf(x): return 1 / (1 + np.exp(-x))

def posTanh(x): return abs(np.tanh(x))

def posSub(x, y): return abs(x-y)

def posTriAdd(x, y, z): return abs(x + y + z)

def TriAdd(x, y, z): return x + y + z

def TriMul(x, y, z): return x * y * z

def posAdd(x, y): return abs(x+y)

def posMul(x, y): return abs(x*y)

def Identity(x): return x

def Neg(x): return -x

def Abs(x): return abs(x)

def Div(left, right):
    try:
        x = left / right
        return x
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError, FloatingPointError, OverflowError):
        return 0.0

def Mul(left, right):
    try:
        #np.seterr(invalid='raise')
        return left * right
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError,
            FloatingPointError, OverflowError):
        return left

def Sqrt(x):
    try:
        if x > 0:
            return np.sqrt(x)
        else:
            return abs(x)
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return 0

def Log(x):
    try:
        if x > 0:
            return np.log(x)
        else:
            return abs(x)
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return 0


def ModExp(x):
    if type(x) == int:
        x = float(x)
    if type(x) != float and type(x) != np.float64:
        out = []
        for i in x:
            if -100<=i<=100:
                out.append(np.exp(i))
            else:
                if i>0:
                    out.append(np.exp(100))
                else:
                    out.append(np.exp(-100))
        return np.array(out)
    else:
        if -100<=x<=100:
            return np.exp(x)
        else:
            if x>0:
                return np.exp(100)
            else:
                return np.exp(-100)

def Sin(x):
    try:
        return np.sin(x)
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return x

def Cos(x):
    try:
        return np.cos(x)
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return x

def PosLog(x):
    try:
        if x > 0:
            return abs(np.log(x))
        else:
            return abs(x)
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return x

def PosSin(x):
    try:
        return abs(np.sin(x))
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return 0

def PosCos(x):
    try:
        return abs(np.cos(x))
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return 0