"""
DeZero Core Simple Module
-------------------------
간단한 버전의 Variable, Function 정의 (MLX 지원)
"""
import weakref
import numpy as np
import contextlib

# =============================================================================
# Optional MLX backend (Apple Silicon GPU)
# =============================================================================
gpu_enable = True
try:
    import mlx.core as mx
except Exception:
    gpu_enable = False
    mx = None


def _is_mlx_array(x):
    return gpu_enable and hasattr(x, "__class__") and getattr(x.__class__, "__module__", "").startswith("mlx.")


def get_array_module(x):
    """Return numpy or mlx module depending on the array type."""
    if isinstance(x, Variable):
        x = x.data
    if x is None:
        return np
    if isinstance(x, np.ndarray) or np.isscalar(x):
        return np
    if _is_mlx_array(x):
        return mx
    return np


def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x


# =============================================================================
# Config
# =============================================================================
class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


# =============================================================================
# Variable / Function
# =============================================================================
class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not (isinstance(data, np.ndarray) or np.isscalar(data) or _is_mlx_array(data)):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            xp = get_array_module(self)
            self.grad = xp.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f is not None and f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        xp = get_array_module(inputs[0]) if inputs else np
        outputs = [Variable(as_array(y, xp)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs]) if inputs else 0
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs):
        raise NotImplementedError()

    def backward(self, *gys):
        raise NotImplementedError()


# =============================================================================
# 사칙연산 / 연산자 오버로드 (NumPy/MLX 양쪽 호환)
# =============================================================================
class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    x0 = as_variable(x0)
    xp = get_array_module(x0)
    x1 = as_array(x1, xp)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0, x1):
    x0 = as_variable(x0)
    xp = get_array_module(x0)
    x1 = as_array(x1, xp)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x0 = as_variable(x0)
    xp = get_array_module(x0)
    x1 = as_array(x1, xp)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_variable(x1)
    xp = get_array_module(x1)
    x0 = as_array(x0, xp)
    return Sub()(x0, x1)


class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / (x1 ** 2))
        return gx0, gx1


def div(x0, x1):
    x0 = as_variable(x0)
    xp = get_array_module(x0)
    x1 = as_array(x1, xp)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_variable(x1)
    xp = get_array_module(x1)
    x0 = as_array(x0, xp)
    return Div()(x0, x1)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        return x ** self.c

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        return c * (x ** (c - 1)) * gy


def pow(x, c):
    return Pow(c)(x)


# Expose operator overloads
def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow


# Quick smoke test
if __name__ == "__main__":
    setup_variable()

    xp = mx if gpu_enable else np

    x = Variable(xp.array(2.0))
    y = x ** 2 + 3 * x + 1
    y.backward()
    print("y = x^2 + 3x + 1, dy/dx should be 2x+3 ->", x.grad)

    a = Variable(xp.array([1.0, 2.0, 3.0]))
    b = Variable(xp.array([4.0, 5.0, 6.0]))
    z = (a * b + a / b - a)
    z.backward()
    print("a.grad:", a.grad)
    print("b.grad:", b.grad)
