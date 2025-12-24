"""
DeZero Transforms Module
------------------------
이미지/배열 변환 함수
"""
import numpy as np
from PIL import Image
from dezero.utils import pair

# MLX 백엔드 shim
try:
    from dezero import cuda as MLX  # ★ 수정
except Exception:
    class MLX:
        @staticmethod
        def get_array_module(x):
            return np


class Compose:
    """Compose several transforms."""
    def __init__(self, transforms=None):
        self.transforms = list(transforms) if transforms else []

    def __call__(self, img):
        if not self.transforms:
            return img
        for t in self.transforms:
            img = t(img)
        return img


# =============================================================================
# Transforms for PIL Image
# =============================================================================
class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, img):
        if self.mode == 'BGR':
            img = img.convert('RGB')
            r, g, b = img.split()
            img = Image.merge('RGB', (b, g, r))
            return img
        else:
            return img.convert(self.mode)


class Resize:
    """Resize the input PIL image to the given size."""
    def __init__(self, size, mode=Image.BILINEAR):
        self.size = pair(size)
        self.mode = mode

    def __call__(self, img):
        return img.resize(self.size, self.mode)


class CenterCrop:
    """Center crop the input PIL image."""
    def __init__(self, size):
        self.size = pair(size)

    def __call__(self, img):
        W, H = img.size
        OW, OH = self.size
        left = (W - OW) // 2
        top = (H - OH) // 2
        right = left + OW
        bottom = top + OH
        return img.crop((left, top, right, bottom))


class ToArray:
    """Convert PIL Image to NumPy/MLX-friendly array (CHW)."""
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return img.astype(self.dtype, copy=False)
        if isinstance(img, Image.Image):
            arr = np.asarray(img)
            if arr.ndim == 2:
                arr = arr[None, :, :]
            else:
                arr = arr.transpose(2, 0, 1)
            return arr.astype(self.dtype, copy=False)
        else:
            raise TypeError("ToArray expects PIL.Image or ndarray.")


class ToPIL:
    """Convert CHW (or HWC) NumPy/MLX array to PIL Image."""
    def __call__(self, array):
        xp = MLX.get_array_module(array)
        arr = xp.asarray(array) if hasattr(xp, 'asarray') else np.asarray(array)

        # MLX를 NumPy로 변환
        if xp is not np:
            arr = np.array(arr)

        if arr.ndim == 3 and (arr.shape[0] in (1, 3, 4)):
            arr = arr.transpose(1, 2, 0)

        if issubclass(arr.dtype.type, np.floating):
            arr = np.clip(arr, 0.0, 1.0) * 255.0
            arr = arr.astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        return Image.fromarray(arr)


class RandomHorizontalFlip:
    """Randomly flip image horizontally with probability p."""
    def __init__(self, p=0.5):
        self.p = float(p)

    def __call__(self, img):
        if np.random.rand() >= self.p:
            return img

        if isinstance(img, Image.Image):
            return img.transpose(Image.FLIP_LEFT_RIGHT)

        xp = MLX.get_array_module(img)
        arr = img

        if arr.ndim == 3 and (arr.shape[0] in (1, 3, 4)):
            axis = 2
        elif arr.ndim == 3:
            axis = 1
        else:
            raise ValueError("RandomHorizontalFlip expects image-like array with 3 dims.")

        flip_fn = getattr(xp, "flip", None)
        if flip_fn is not None:
            return flip_fn(arr, axis=axis)
        else:
            idx = np.arange(arr.shape[axis] - 1, -1, -1)
            take_fn = getattr(xp, "take", None)
            if take_fn is not None:
                return take_fn(arr, idx, axis=axis)
            slicer = [slice(None)] * arr.ndim
            slicer[axis] = slice(None, None, -1)
            return arr[tuple(slicer)]


# =============================================================================
# Transforms for NumPy/MLX ndarray
# =============================================================================
class Normalize:
    """Normalize an array with mean and std."""
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        xp = MLX.get_array_module(array)
        mean, std = self.mean, self.std

        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(self.mean) == 1 else len(self.mean)
            mean = xp.array(self.mean, dtype=array.dtype).reshape(*mshape)
        if not np.isscalar(std):
            rshape = [1] * array.ndim
            rshape[0] = len(array) if len(self.std) == 1 else len(self.std)
            std = xp.array(self.std, dtype=array.dtype).reshape(*rshape)

        return (array - mean) / std


class Flatten:
    """Flatten an array."""
    def __call__(self, array):
        return array.reshape(-1)


class AsType:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.astype(self.dtype, copy=False)


ToFloat = AsType


class ToInt(AsType):
    def __init__(self, dtype=int):
        self.dtype = dtype
