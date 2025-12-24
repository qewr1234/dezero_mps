"""
DeZero CUDA/MLX Backend Shim
----------------------------
macOS에서는 CuPy 대신 MLX를 사용합니다.
"""
import numpy as np

# MLX 사용 가능 여부 확인
gpu_enable = False
mx = None

try:
    import mlx.core as mx
    gpu_enable = True
except ImportError:
    pass


def get_array_module(x):
    """입력 데이터가 MLX 배열인지 NumPy 배열인지 판단하여 모듈 반환"""
    if x is None:
        return np
    
    if isinstance(x, (list, tuple)):
        if len(x) > 0:
            x = x[0]
        else:
            return np
    
    # DeZero Variable인 경우 data를 꺼내서 확인
    if hasattr(x, 'data'):
        x = x.data
    
    if x is None:
        return np

    if gpu_enable and mx is not None and isinstance(x, mx.array):
        return mx
    
    return np


def as_numpy(x):
    """MLX 배열이든 무엇이든 NumPy 배열로 강제 변환"""
    if x is None:
        return None
        
    if isinstance(x, (list, tuple)):
        return np.array(x)
    
    if hasattr(x, 'data'):
        x = x.data

    if x is None:
        return None

    if np.isscalar(x):
        return np.array(x)
    
    if isinstance(x, np.ndarray):
        return x
    
    if gpu_enable and mx is not None and isinstance(x, mx.array):
        return np.array(x)  # MLX -> NumPy 변환
        
    return np.array(x)


def as_mlx(x):
    """NumPy 배열을 MLX 배열(GPU)로 변환"""
    if not gpu_enable or mx is None:
        return as_numpy(x)
        
    if hasattr(x, 'data'):
        x = x.data

    if x is None:
        return None

    if isinstance(x, mx.array):
        return x
        
    return mx.array(x)


# 호환성을 위한 alias
as_cupy = as_mlx
as_gpu = as_mlx


def asarray(x):
    """현재 기본 백엔드로 배열 변환 (gradient_check용)"""
    if gpu_enable and mx is not None:
        return mx.array(x) if not isinstance(x, mx.array) else x
    return np.asarray(x)
