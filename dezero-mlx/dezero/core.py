"""
DeZero Core Module
------------------
Variable, Parameter, Function 등 핵심 클래스 정의
"""
import weakref
import numpy as np
import contextlib

# =============================================================================
# Config
# =============================================================================
class Config:
    enable_backprop = True
    train = True


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


def test_mode():
    return using_config('train', False)


# =============================================================================
# Variable / Parameter / Function
# =============================================================================
try:
    import mlx.core as mx
    array_types = (np.ndarray, mx.array)
except ImportError:
    array_types = (np.ndarray,)
    mx = None


def _get_array_module(x):
    """내부용: cuda 모듈 import 없이 배열 모듈 판단"""
    if x is None:
        return np
    if hasattr(x, 'data'):
        x = x.data
    if x is None:
        return np
    if mx is not None and isinstance(x, mx.array):
        return mx
    return np


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, array_types):
                if np.isscalar(data):
                    data = np.array(data)
                else:
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

    def unchain(self):
        self.creator = None

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            xp = _get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output is weakref

            with using_config('enable_backprop', create_graph):
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
                    y().grad = None  # y is weakref

    def unchain_backward(self):
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        from dezero import functions
        return functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        from dezero import functions
        return functions.transpose(self, axes)

    @property
    def T(self):
        from dezero import functions
        return functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        from dezero import functions
        return functions.sum(self, axis, keepdims)

    def to_cpu(self):
        from dezero import cuda
        if self.data is not None:
            self.data = cuda.as_numpy(self.data)

    def to_gpu(self):
        from dezero import cuda
        if self.data is not None:
            self.data = cuda.as_gpu(self.data)


class Parameter(Variable):
    pass


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


# =============================================================================
# 사칙연산 / 연산자 오버로드
# =============================================================================
class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        from dezero import functions
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = functions.sum_to(gx0, self.x0_shape)
            gx1 = functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0, x1):
    from dezero import cuda
    x1 = as_array(x1, cuda.get_array_module(x0.data))
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        from dezero import functions
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:
            gx0 = functions.sum_to(gx0, x0.shape)
            gx1 = functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0, x1):
    from dezero import cuda
    x1 = as_array(x1, cuda.get_array_module(x0.data))
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
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        from dezero import functions
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:
            gx0 = functions.sum_to(gx0, self.x0_shape)
            gx1 = functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0, x1):
    from dezero import cuda
    x1 = as_array(x1, cuda.get_array_module(x0.data))
    return Sub()(x0, x1)


def rsub(x0, x1):
    from dezero import cuda
    x1 = as_array(x1, cuda.get_array_module(x0.data))
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        from dezero import functions
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:
            gx0 = functions.sum_to(gx0, x0.shape)
            gx1 = functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def div(x0, x1):
    from dezero import cuda
    x1 = as_array(x1, cuda.get_array_module(x0.data))
    return Div()(x0, x1)


def rdiv(x0, x1):
    from dezero import cuda
    x1 = as_array(x1, cuda.get_array_module(x0.data))
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


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
    
    # 지연 import로 순환 참조 방지
    def _getitem(self, slices):
        from dezero import functions
        return functions.get_item(self, slices)
    
    def _matmul(self, other):
        from dezero import functions
        return functions.matmul(self, other)
    
    def _max(self, axis=None, keepdims=False):
        from dezero import functions
        return functions.max(self, axis, keepdims)
    
    def _min(self, axis=None, keepdims=False):
        from dezero import functions
        return functions.min(self, axis, keepdims)
    
    Variable.__getitem__ = _getitem
    Variable.matmul = _matmul
    Variable.dot = _matmul
    Variable.max = _max
    Variable.min = _min
