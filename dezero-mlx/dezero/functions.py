"""
DeZero Functions Module
-----------------------
활성화 함수, 손실 함수, 변환 함수 등
"""
import numpy as np
from dezero.core import Function, Variable, as_variable, as_array
from dezero import cuda
from dezero import utils


# =============================================================================
# Basic functions: sin / cos / tanh / exp / log
# =============================================================================
class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)


class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


# =============================================================================
# Tensor operations: reshape / transpose / get_item / expand_dims / flatten
# =============================================================================
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)
        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            # MLX: scatter_add가 없으면 인덱싱으로 처리
            gx = gx.at[self.slices].add(gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    return GetItem(slices)(x)


def expand_dims(x, axis):
    x = as_variable(x)
    shape = list(x.shape)
    shape.insert(axis, 1)
    return reshape(x, tuple(shape))


def flatten(x):
    """Flattens the input. Does not affect the batch size."""
    return reshape(x, (x.shape[0], -1))


# =============================================================================
# sum / sum_to / broadcast_to / average / matmul / linear
# =============================================================================
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        xp = cuda.get_array_module(x)
        y = xp.sum(x, axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis,
                                        self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        xp = cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


def average(x, axis=None, keepdims=False):
    x = as_variable(x)
    y = sum(x, axis, keepdims)
    return y * (y.data.size / x.data.size)


mean = average


class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)


class BatchMatMul(Function):
    """Batch Matrix Multiplication for 3D tensors.
    
    Supports:
    - (batch, n, m) @ (batch, m, k) -> (batch, n, k)
    - Broadcasting is supported
    """
    def forward(self, A, B):
        xp = cuda.get_array_module(A)
        # numpy/mlx의 matmul은 batch dimension을 자동 처리
        y = xp.matmul(A, B)
        return y

    def backward(self, gy):
        A, B = self.inputs
        # dL/dA = dL/dY @ B^T
        # dL/dB = A^T @ dL/dY
        gA = batch_matmul(gy, B.transpose(0, 2, 1))
        gB = batch_matmul(A.transpose(0, 2, 1), gy)
        return gA, gB


def batch_matmul(A, B):
    """Batch matrix multiplication for 3D tensors."""
    return BatchMatMul()(A, B)


class Linear(Function):
    def forward(self, x, W, b):
        # 3D 텐서 지원: (batch, seq, features) -> (batch*seq, features)
        self.x_shape = x.shape
        if x.ndim == 3:
            batch, seq, features = x.shape
            x = x.reshape(batch * seq, features)
        
        y = x.dot(W)
        if b is not None:
            y += b
        
        # 다시 3D로 복원
        if len(self.x_shape) == 3:
            batch, seq, _ = self.x_shape
            y = y.reshape(batch, seq, -1)
        
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        
        # 3D면 2D로 변환
        original_gy_shape = gy.shape
        if gy.ndim == 3:
            batch, seq, features = gy.shape
            gy_2d = gy.reshape(batch * seq, features)
        else:
            gy_2d = gy
        
        if x.ndim == 3:
            batch, seq, features = x.shape
            x_2d = x.reshape(batch * seq, features)
        else:
            x_2d = x
        
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy_2d, W.T)
        gW = matmul(x_2d.T, gy_2d)
        
        # gx를 원래 shape로 복원
        if len(self.x_shape) == 3:
            batch, seq, features = self.x_shape
            gx = gx.reshape(batch, seq, features)
        
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


# =============================================================================
# Activation functions: sigmoid / relu / softmax / log_softmax / leaky_relu
# =============================================================================
class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x * 0.5) * 0.5 + 0.5
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)


class ReLU(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x):
    return ReLU()(x)


class LeakyReLU(Function):
    def __init__(self, slope):
        self.slope = slope

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.where(x >= 0, x, x * self.slope)
        return y

    def backward(self, gy):
        x, = self.inputs
        xp = cuda.get_array_module(x.data)
        mask = xp.where(x.data >= 0, 1, self.slope)
        gx = gy * mask
        return gx


def leaky_relu(x, slope=0.2):
    return LeakyReLU(slope)(x)


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)


class LogSoftmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        log_z = utils.logsumexp(x, self.axis)
        y = x - log_z
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy - exp(y) * gy.sum(axis=self.axis, keepdims=True)
        return gx


def log_softmax(x, axis=1):
    return LogSoftmax(axis)(x)


# =============================================================================
# Loss functions: mean_squared_error / softmax_cross_entropy / sigmoid_cross_entropy
# =============================================================================
class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        xp = cuda.get_array_module(x)
        log_p = log_p[xp.arange(N), t.ravel()]
        y = -log_p.sum() / N
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1 / N
        y = softmax(x)
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


def sigmoid_cross_entropy(x, t):
    if x.ndim != t.ndim:
        t = t.reshape(*x.shape)
    x, t = as_variable(x), as_variable(t)
    N = len(x)
    p = sigmoid(x)
    p = clip(p, 1e-15, 1.0)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y


def binary_cross_entropy(p, t):
    if p.ndim != t.ndim:
        t = t.reshape(*p.shape)
    N = len(t)
    p = clip(p, 1e-15, 0.999)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y


# =============================================================================
# Max / Min / Clip
# =============================================================================
class Max(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.max(x, axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()

        shape = utils.max_backward_shape(x, self.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond = (x.data == y.data)
        gy = broadcast_to(gy, cond.shape)
        return gy * cond


class Min(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.min(x, axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()

        shape = utils.max_backward_shape(x, self.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond = (x.data == y.data)
        gy = broadcast_to(gy, cond.shape)
        return gy * cond


def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)


def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


# =============================================================================
# accuracy / dropout / batch_norm
# =============================================================================
def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)
    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))


def dropout(x, dropout_ratio=0.5):
    """Dropout regularization.
    
    Supports any shape tensor (1D, 2D, 3D, etc.)
    """
    from dezero import Config
    x = as_variable(x)

    if not Config.train:
        return x
    
    xp = cuda.get_array_module(x.data)
    mask = xp.random.rand(*x.shape) > dropout_ratio
    scale = xp.array(1.0 / (1.0 - dropout_ratio)).astype(x.dtype)
    y = x * Variable(mask.astype(x.dtype)) * scale
    return y


class BatchNorm(Function):
    def __init__(self, mean, var, decay, eps):
        self.avg_mean = mean
        self.avg_var = var
        self.decay = decay
        self.eps = eps
        self.inv_std = None

    def forward(self, x, gamma, beta):
        from dezero import Config
        assert x.ndim == 2 or x.ndim == 4

        x_ndim = x.ndim
        if x_ndim == 4:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)

        xp = cuda.get_array_module(x)

        if Config.train:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            inv_std = 1 / xp.sqrt(var + self.eps)
            xc = (x - mean) * inv_std

            m = x.size // gamma.size
            s = m - 1. if m - 1. > 1. else 1.
            adjust = m / s
            self.avg_mean *= self.decay
            self.avg_mean += (1 - self.decay) * mean
            self.avg_var *= self.decay
            self.avg_var += (1 - self.decay) * adjust * var
            self.inv_std = inv_std
        else:
            inv_std = 1 / xp.sqrt(self.avg_var + self.eps)
            xc = (x - self.avg_mean) * inv_std

        y = gamma * xc + beta

        if x_ndim == 4:
            y = y.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return y

    def backward(self, gy):
        gy_ndim = gy.ndim
        if gy_ndim == 4:
            N, C, H, W = gy.shape
            gy = gy.transpose(0, 2, 3, 1).reshape(-1, C)

        x, gamma, beta = self.inputs
        batch_size = len(gy)

        if x.ndim == 4:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)
        mean = x.sum(axis=0) / batch_size
        xc = (x - mean) * self.inv_std

        gbeta = sum(gy, axis=0)
        ggamma = sum(xc * gy, axis=0)
        gx = gy - gbeta / batch_size - xc * ggamma / batch_size
        gx *= gamma * self.inv_std

        if gy_ndim == 4:
            gx = gx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return gx, ggamma, gbeta
    
class EmbedIDFunction(Function):
    def forward(self, W, x):
        self.x_data = x
        y = W[x]
        return y

    def backward(self, gy):
        W, _ = self.inputs
        xp = cuda.get_array_module(gy.data)
        gW = xp.zeros_like(W.data)
        
        x_flat = self.x_data.flatten()
        gy_flat = gy.data.reshape(-1, gy.shape[-1])
        
        # NumPy/MLX 호환
        if hasattr(xp, 'add') and hasattr(xp.add, 'at'):
            xp.add.at(gW, x_flat, gy_flat)
        else:
            for i, idx in enumerate(x_flat):
                gW[idx] = gW[idx] + gy_flat[i]
        
        return Variable(gW), None
    
def embed_id(x, W):
    """Embedding lookup."""
    return EmbedIDFunction()(W, x)


def batch_norm(x, gamma, beta, mean, var, decay=0.9, eps=2e-5):
    return BatchNorm(mean, var, decay, eps)(x, gamma, beta)


# =============================================================================
# Placeholder for conv functions (defined in functions_conv.py)
# =============================================================================
def conv2d(x, W, b=None, stride=1, pad=0):
    from dezero import functions_conv
    return functions_conv.conv2d(x, W, b, stride, pad)


def deconv2d(x, W, b=None, stride=1, pad=0, outsize=None):
    from dezero import functions_conv
    return functions_conv.deconv2d(x, W, b, stride, pad, outsize)


def pooling(x, kernel_size, stride=1, pad=0):
    from dezero import functions_conv
    return functions_conv.pooling(x, kernel_size, stride, pad)


def average_pooling(x, kernel_size, stride=1, pad=0):
    from dezero import functions_conv
    return functions_conv.average_pooling(x, kernel_size, stride, pad)