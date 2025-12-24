# dezero/__init__.py
"""
DeZero: 밑바닥부터 시작하는 딥러닝 프레임워크 (macOS MLX 지원)
"""

is_simple_core = False

if is_simple_core:
    from .core_simple import Variable
    from .core_simple import Function
    from .core_simple import using_config
    from .core_simple import no_grad
    from .core_simple import as_array
    from .core_simple import as_variable
    from .core_simple import setup_variable
else:
    from .core import Variable, Parameter, Function, using_config, no_grad
    from .core import test_mode, as_array, as_variable, setup_variable, Config
    from .layers import Layer
    from .models import Model
    from .datasets import Dataset
    from .dataloaders import DataLoader, SeqDataLoader

    from . import datasets
    from . import dataloaders
    from . import optimizers
    from . import functions
    from . import functions_conv
    from . import layers
    from . import utils
    from . import transforms
    from . import cuda

    # ★ 핵심 수정: MLX를 cuda의 alias로 설정
    # 다른 모듈들이 `from . import MLX`로 import할 때 사용됨
    MLX = cuda

setup_variable()
__version__ = '0.0.13'

# 깔끔한 공개 심볼
__all__ = [
    # core
    'Variable', 'Parameter', 'Function', 'using_config', 'no_grad', 'test_mode',
    'as_array', 'as_variable', 'setup_variable', 'Config',
    # high level
    'Layer', 'Model', 'Dataset', 'DataLoader', 'SeqDataLoader',
    # submodules
    'datasets', 'dataloaders', 'optimizers', 'functions', 'functions_conv',
    'layers', 'utils', 'transforms', 'cuda', 'MLX',
]
