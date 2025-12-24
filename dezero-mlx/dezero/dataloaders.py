"""
DeZero DataLoaders
------------------
NumPy / MLX 친화 버전
"""
from __future__ import annotations
import math
import numpy as np
from typing import Any, Sequence

# 백엔드 선택
try:
    from dezero import cuda  # ★ 수정: 상대 경로가 아닌 패키지 import
except Exception:
    cuda = None

try:
    import mlx.core as mx
except Exception:
    mx = None


def _select_xp(use_gpu: bool):
    """현재 설정에서 사용할 array 모듈(np or mx)을 선택"""
    if use_gpu:
        if cuda is not None:
            backend_mx = getattr(cuda, "mx", None)
            if backend_mx is not None:
                return backend_mx
        if mx is not None:
            return mx
    return np


def _to_xp_batch(xp, items: Sequence[Any]):
    """파이썬/NumPy 리스트를 대상 백엔드 배열로 변환"""
    if xp is np:
        return np.asarray(items)
    try:
        return xp.array(items)
    except Exception:
        arr = np.asarray(items)
        return xp.array(arr)


class DataLoader:
    def __init__(self, dataset, batch_size: int, shuffle: bool = True, gpu: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / self.batch_size)
        self.gpu = bool(gpu)
        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(self.data_size)
        else:
            self.index = np.arange(self.data_size)

    def __len__(self):
        return self.max_iter

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i = self.iteration
        B = self.batch_size
        idx = self.index[i * B:(i + 1) * B]
        batch = [self.dataset[j] for j in idx]

        xp = _select_xp(self.gpu)
        xs = [ex[0] for ex in batch]
        ts = [ex[1] for ex in batch]

        x = _to_xp_batch(xp, xs)
        t = None if any(ti is None for ti in ts) else _to_xp_batch(xp, ts)

        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()

    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        self.gpu = True


class SeqDataLoader(DataLoader):
    """RNN/LangTask용 시퀀스 배치 로더."""
    def __init__(self, dataset, batch_size: int, gpu: bool = False):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False, gpu=gpu)

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        jump = self.data_size // self.batch_size
        idx = [(i * jump + self.iteration) % self.data_size for i in range(self.batch_size)]
        batch = [self.dataset[j] for j in idx]

        xp = _select_xp(self.gpu)
        xs = [ex[0] for ex in batch]
        ts = [ex[1] for ex in batch]

        x = _to_xp_batch(xp, xs)
        t = None if any(ti is None for ti in ts) else _to_xp_batch(xp, ts)

        self.iteration += 1
        return x, t
