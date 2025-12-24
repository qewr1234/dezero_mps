#!/usr/bin/env python3
"""
DeZero 종합 테스트 스크립트
macOS MLX 지원 버전
"""
import sys
sys.path.append('/Users/dbwns/프로그래밍/DeepLearning')

import numpy as np

# =============================================================================
# 1. 기본 Import 테스트
# =============================================================================
print("=" * 60)
print("🍎 DeZero macOS MLX 버전 테스트")
print("=" * 60)

import dezero
from dezero import Variable, Parameter
from dezero import functions as F
from dezero import layers as L
from dezero import optimizers
from dezero import cuda
from dezero.models import MLP, Sequential
from dezero.datasets import Spiral
from dezero.dataloaders import DataLoader

print(f"\n✓ DeZero v{dezero.__version__} 로드 완료")
print(f"✓ GPU(MLX) 사용 가능: {cuda.gpu_enable}")

# =============================================================================
# 2. 자동 미분 테스트
# =============================================================================
print("\n" + "=" * 60)
print("📐 자동 미분 테스트")
print("=" * 60)

# 간단한 함수: y = x^2 + 2x + 1
x = Variable(np.array(3.0))
y = x ** 2 + 2 * x + 1
y.backward()

print(f"\ny = x² + 2x + 1")
print(f"x = {x.data}")
print(f"y = {y.data}")
print(f"dy/dx = {x.grad.data} (정답: 2*3+2 = 8)")

# 복잡한 함수: z = sin(x) + cos(x)
x = Variable(np.array(np.pi / 4))
z = F.sin(x) + F.cos(x)
z.backward()

print(f"\nz = sin(x) + cos(x)")
print(f"x = π/4 = {x.data:.4f}")
print(f"z = {z.data:.4f} (정답: √2 ≈ 1.414)")
print(f"dz/dx = {x.grad.data:.4f} (정답: cos(π/4) - sin(π/4) = 0)")

# =============================================================================
# 3. 신경망 레이어 테스트
# =============================================================================
print("\n" + "=" * 60)
print("🧠 신경망 레이어 테스트")
print("=" * 60)

# Linear 레이어
linear = L.Linear(10, in_size=5)
x = Variable(np.random.randn(4, 5).astype(np.float32))
y = linear(x)
print(f"\nLinear(5 → 10)")
print(f"  입력: {x.shape} → 출력: {y.shape}")

# Sequential 모델
model = Sequential(
    L.Linear(64),
    L.Linear(32),
    L.Linear(10)
)
x = Variable(np.random.randn(8, 100).astype(np.float32))
y = F.relu(model.layers[0](x))
y = F.relu(model.layers[1](y))
y = model.layers[2](y)
print(f"\nSequential(100 → 64 → 32 → 10)")
print(f"  입력: (8, 100) → 출력: {y.shape}")

# =============================================================================
# 4. Spiral 데이터셋 학습 테스트
# =============================================================================
print("\n" + "=" * 60)
print("🌀 Spiral 데이터셋 학습 테스트")
print("=" * 60)

# 하이퍼파라미터
max_epoch = 50
batch_size = 30
hidden_size = 10
lr = 1.0

# 데이터 로드
train_set = Spiral(train=True)
test_set = Spiral(train=False)
train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

print(f"\n학습 데이터: {len(train_set)}개")
print(f"테스트 데이터: {len(test_set)}개")

# 모델 & 옵티마이저
model = MLP((hidden_size, 3), activation=F.relu)
optimizer = optimizers.SGD(lr).setup(model)

# 학습 루프
print(f"\n학습 시작 (epochs: {max_epoch})")
print("-" * 40)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0
    
    for x, t in train_loader:
        x = Variable(x)
        t = Variable(t)
        
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        
        model.cleargrads()
        loss.backward()
        optimizer.update()
        
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)
    
    avg_loss = sum_loss / len(train_set)
    avg_acc = sum_acc / len(train_set)
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

# 테스트
print("-" * 40)
print("테스트 중...")

with dezero.no_grad():
    sum_acc = 0
    for x, t in test_loader:
        x = Variable(x)
        t = Variable(t)
        y = model(x)
        acc = F.accuracy(y, t)
        sum_acc += float(acc.data) * len(t)
    
    test_acc = sum_acc / len(test_set)
    print(f"테스트 정확도: {test_acc:.4f}")

# =============================================================================
# 5. 옵티마이저 비교 테스트
# =============================================================================
print("\n" + "=" * 60)
print("⚡ 옵티마이저 비교 테스트")
print("=" * 60)

def train_with_optimizer(opt_class, opt_name, **kwargs):
    model = MLP((hidden_size, 3), activation=F.relu)
    optimizer = opt_class(**kwargs).setup(model)
    
    losses = []
    for epoch in range(30):
        sum_loss = 0
        for x, t in train_loader:
            x, t = Variable(x), Variable(t)
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data)
        losses.append(sum_loss / len(train_loader))
    
    print(f"  {opt_name:15s} | 최종 Loss: {losses[-1]:.4f}")
    return losses

print("\n30 에폭 학습 후 Loss 비교:")
print("-" * 40)

train_with_optimizer(optimizers.SGD, "SGD", lr=1.0)
train_with_optimizer(optimizers.MomentumSGD, "MomentumSGD", lr=0.1)
train_with_optimizer(optimizers.AdaGrad, "AdaGrad", lr=0.1)
train_with_optimizer(optimizers.Adam, "Adam", alpha=0.01)

# =============================================================================
# 6. GPU (MLX) 테스트 (가능한 경우)
# =============================================================================
print("\n" + "=" * 60)
print("🚀 GPU (MLX) 테스트")
print("=" * 60)

if cuda.gpu_enable:
    import mlx.core as mx
    
    # NumPy → MLX 변환
    x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x_mlx = cuda.as_gpu(x_np)
    
    print(f"\nNumPy 배열: {x_np}")
    print(f"MLX 배열: {x_mlx}")
    print(f"타입: {type(x_mlx)}")
    
    # MLX에서 연산
    y_mlx = x_mlx ** 2 + x_mlx
    print(f"x² + x = {y_mlx}")
    
    # 다시 NumPy로
    y_np = cuda.as_numpy(y_mlx)
    print(f"NumPy로 변환: {y_np}")
    
    print("\n✓ MLX 백엔드 정상 작동!")
else:
    print("\n⚠ MLX가 설치되어 있지 않습니다.")
    print("  Apple Silicon Mac에서 MLX를 설치하려면:")
    print("  $ pip install mlx")

# =============================================================================
# 완료
# =============================================================================
print("\n" + "=" * 60)
print("✅ 모든 테스트 완료!")
print("=" * 60)
