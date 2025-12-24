"""
DeZero Convolution Functions
----------------------------
im2col, col2im, conv2d, deconv2d, pooling 등
"""
import numpy as np
from dezero.core import Function, as_variable
from dezero import cuda
from dezero.utils import pair, get_conv_outsize, get_deconv_outsize


# =============================================================================
# im2col / col2im
# =============================================================================
def im2col_array(img, kernel_size, stride, pad, to_matrix=True):
    """이미지를 컬럼 배열로 변환 (NumPy/MLX 호환)"""
    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    xp = cuda.get_array_module(img)

    if xp is np:
        img = np.pad(img, ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
                     mode='constant', constant_values=(0,))
    else:
        # MLX padding
        img = xp.pad(img, ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)))

    col = xp.zeros((N, C, KH, KW, OH, OW), dtype=img.dtype)

    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OW
            col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

    return col


def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    """컬럼 배열을 이미지로 변환 (NumPy/MLX 호환)"""
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    xp = cuda.get_array_module(col)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose((0, 3, 4, 5, 1, 2))

    img = xp.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1), dtype=col.dtype)

    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OW
            if xp is np:
                np.add.at(img, (slice(None), slice(None),
                               slice(j, j_lim, SH), slice(i, i_lim, SW)), col[:, :, j, i, :, :])
            else:
                # MLX: at indexer 사용
                img = img.at[:, :, j:j_lim:SH, i:i_lim:SW].add(col[:, :, j, i, :, :])

    return img[:, :, PH:H + PH, PW:W + PW]


class Im2col(Function):
    def __init__(self, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        self.input_shape = x.shape
        y = im2col_array(x, self.kernel_size, self.stride, self.pad,
                         self.to_matrix)
        return y

    def backward(self, gy):
        gx = col2im(gy, self.input_shape, self.kernel_size, self.stride,
                    self.pad, self.to_matrix)
        return gx


def im2col(x, kernel_size, stride=1, pad=0, to_matrix=True):
    return Im2col(kernel_size, stride, pad, to_matrix)(x)


class Col2im(Function):
    def __init__(self, input_shape, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        y = col2im_array(x, self.input_shape, self.kernel_size, self.stride,
                         self.pad, self.to_matrix)
        return y

    def backward(self, gy):
        gx = im2col(gy, self.kernel_size, self.stride, self.pad,
                    self.to_matrix)
        return gx


def col2im(x, input_shape, kernel_size, stride=1, pad=0, to_matrix=True):
    return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)


# =============================================================================
# Conv2d / Deconv2d
# =============================================================================
class Conv2d(Function):
    def __init__(self, stride=1, pad=0):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)

    def forward(self, x, W, b):
        xp = cuda.get_array_module(x)

        KH, KW = W.shape[2:]
        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=True)

        y = xp.dot(col, W.reshape(W.shape[0], -1).T)
        if b is not None:
            y += b
        y = y.reshape(*x.shape[:1], *y.shape[1:])

        N = x.shape[0]
        OH = get_conv_outsize(x.shape[2], KH, self.stride[0], self.pad[0])
        OW = get_conv_outsize(x.shape[3], KW, self.stride[1], self.pad[1])
        y = y.reshape(N, OH, OW, -1).transpose(0, 3, 1, 2)

        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gx = deconv2d(gy, W, b=None, stride=self.stride, pad=self.pad,
                      outsize=(x.shape[2], x.shape[3]))
        gW = Conv2dGradW(self)(x, gy)
        gb = None
        if b.data is not None:
            from dezero import functions
            gb = functions.sum_to(gy, b.shape)
        return gx, gW, gb


def conv2d(x, W, b=None, stride=1, pad=0):
    return Conv2d(stride, pad)(x, W, b)


class Deconv2d(Function):
    def __init__(self, stride=1, pad=0, outsize=None):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.outsize = outsize

    def forward(self, x, W, b):
        xp = cuda.get_array_module(x)

        Weight = W
        SH, SW = self.stride
        PH, PW = self.pad
        C, OC, KH, KW = Weight.shape
        N, C, H, W = x.shape
        if self.outsize is None:
            out_h = get_deconv_outsize(H, KH, SH, PH)
            out_w = get_deconv_outsize(W, KW, SW, PW)
        else:
            out_h, out_w = pair(self.outsize)
        img_shape = (N, OC, out_h, out_w)

        gcol = xp.dot(x.reshape(N * H * W, C), Weight.reshape(C, -1))
        gcol = gcol.reshape(N, H, W, OC, KH, KW).transpose(0, 3, 4, 5, 1, 2)
        y = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad,
                         to_matrix=False)
        if b is not None:
            y += b.reshape((1, b.size, 1, 1))
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gx = conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)
        gW = Conv2dGradW(self)(gy, x)
        gb = None
        if b.data is not None:
            from dezero import functions
            gb = functions.sum_to(gy, b.shape)
        return gx, gW, gb


def deconv2d(x, W, b=None, stride=1, pad=0, outsize=None):
    return Deconv2d(stride, pad, outsize)(x, W, b)


class Conv2dGradW(Function):
    def __init__(self, conv2d):
        W = conv2d.inputs[1]
        kh, kw = W.shape[2:]
        self.kernel_size = (kh, kw)
        self.stride = conv2d.stride
        self.pad = conv2d.pad

    def forward(self, x, gy):
        xp = cuda.get_array_module(x)

        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=True)
        gy = gy.transpose(0, 2, 3, 1).reshape(-1, gy.shape[1])

        gW = xp.dot(gy.T, col)
        gW = gW.reshape(gy.shape[1], x.shape[1], *self.kernel_size)
        return gW

    def backward(self, gys):
        x, gy = self.inputs
        gx = deconv2d(gy, gys, stride=self.stride, pad=self.pad)
        ggy = conv2d(x, gys, stride=self.stride, pad=self.pad)
        return gx, ggy


# =============================================================================
# Pooling
# =============================================================================
class Pooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        xp = cuda.get_array_module(x)
        N, C, H, W = x.shape
        KH, KW = pair(self.kernel_size)
        PH, PW = pair(self.pad)
        SH, SW = pair(self.stride)
        OH = get_conv_outsize(H, KH, SH, PH)
        OW = get_conv_outsize(W, KW, SW, PW)

        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=True)
        col = col.reshape(-1, KH * KW)
        self.indexes = col.argmax(axis=1)
        y = col.max(axis=1)
        y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
        return y

    def backward(self, gy):
        return Pooling2DGrad(self)(gy)


class Pooling2DGrad(Function):
    def __init__(self, mpool2d):
        self.mpool2d = mpool2d
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, gy):
        xp = cuda.get_array_module(gy)
        N, C, OH, OW = gy.shape
        N, C, H, W = self.input_shape
        KH, KW = pair(self.kernel_size)

        gy = gy.transpose(0, 2, 3, 1).flatten()
        
        # One-hot encoding for max locations
        if xp is np:
            col = np.zeros((gy.size, KH * KW), dtype=self.dtype)
            col[np.arange(gy.size), self.indexes.flatten()] = gy
        else:
            col = xp.zeros((gy.size, KH * KW), dtype=self.dtype)
            # MLX에서는 at indexer 사용
            indices = xp.arange(gy.size)
            col = col.at[indices, self.indexes.flatten()].set(gy)
        
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
        gx = col2im_array(col, (N, C, H, W), self.kernel_size, self.stride,
                          self.pad, to_matrix=False)
        return gx

    def backward(self, ggx):
        f = Pooling2DWithIndexes(self.mpool2d)
        return f(ggx)


class Pooling2DWithIndexes(Function):
    def __init__(self, mpool2d):
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, x):
        xp = cuda.get_array_module(x)
        N, C, H, W = x.shape
        KH, KW = pair(self.kernel_size)
        SH, SW = pair(self.stride)
        PH, PW = pair(self.pad)
        OH = get_conv_outsize(H, KH, SH, PH)
        OW = get_conv_outsize(W, KW, SW, PW)

        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=True)
        col = col.reshape(-1, KH * KW)
        
        if xp is np:
            y = col[np.arange(len(col)), self.indexes.flatten()]
        else:
            # MLX
            y = col[xp.arange(len(col)), self.indexes.flatten()]
        
        y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
        return y

    def backward(self, gy):
        return Pooling2DGrad(self)(gy)


def pooling(x, kernel_size, stride=1, pad=0):
    return Pooling(kernel_size, stride, pad)(x)


# Alias
max_pooling = pooling


class AveragePooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        KH, KW = pair(self.kernel_size)
        PH, PW = pair(self.pad)
        SH, SW = pair(self.stride)
        OH = get_conv_outsize(H, KH, SH, PH)
        OW = get_conv_outsize(W, KW, SW, PW)

        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=True)
        col = col.reshape(-1, KH * KW)
        y = col.mean(axis=1)
        y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
        return y

    def backward(self, gy):
        N, C, OH, OW = gy.shape
        KH, KW = pair(self.kernel_size)
        gy /= (KH * KW)
        gy = gy.transpose(0, 2, 3, 1).reshape(-1)

        xp = cuda.get_array_module(gy.data)
        
        # Broadcast gradient
        gcol = xp.broadcast_to(gy.data.reshape(-1, 1), (gy.size, KH * KW))
        gcol = gcol.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

        N, C, H, W = self.inputs[0].shape
        gx = col2im_array(gcol, (N, C, H, W), self.kernel_size, self.stride,
                          self.pad, to_matrix=False)
        return as_variable(gx)


def average_pooling(x, kernel_size, stride=1, pad=0):
    return AveragePooling(kernel_size, stride, pad)(x)
