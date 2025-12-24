# 🍎 DeZero-MLX

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Platform-macOS-lightgrey.svg" alt="macOS">
  <img src="https://img.shields.io/badge/Backend-MLX%20%7C%20NumPy-orange.svg" alt="Backend">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

<p align="center">
  <b>Deep Learning Framework from Scratch with Apple Silicon GPU Support</b>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-examples">Examples</a> •
  <a href="#한국어">한국어</a>
</p>

---

**DeZero-MLX** is a deep learning framework built from scratch, modified to support Apple Silicon GPUs via [MLX](https://github.com/ml-explore/mlx). Based on the original [DeZero](https://github.com/oreilly-japan/deep-learning-from-scratch-3) framework from the book "Deep Learning from Scratch 3".

## ✨ Features

- 🚀 **Apple Silicon GPU Acceleration** - Seamless MLX backend support
- 🔄 **Automatic Differentiation** - Define-by-run dynamic computation graph
- 🧠 **Neural Network Layers** - Linear, Conv2d, LSTM, BatchNorm, and more
- ⚡ **Optimizers** - SGD, Momentum, AdaGrad, Adam
- 📊 **Built-in Datasets** - MNIST, Spiral, SinCurve
- 🔧 **NumPy Fallback** - Works on any platform without MLX

## 📦 Installation

### Requirements

- Python 3.8+
- NumPy
- (Optional) MLX for Apple Silicon GPU support

### Install

```bash
# Clone the repository
git clone https://github.com/yourusername/dezero-mlx.git
cd dezero-mlx

# Install dependencies
pip install numpy

# (Optional) Install MLX for GPU acceleration on Apple Silicon
pip install mlx
```

## 🚀 Quick Start

```python
import numpy as np
from dezero import Variable
import dezero.functions as F

# Automatic differentiation
x = Variable(np.array(2.0))
y = x ** 2 + 3 * x + 1
y.backward()

print(f"y = {y.data}")      # y = 11.0
print(f"dy/dx = {x.grad}")  # dy/dx = 7.0
```

## 📚 Examples

### Neural Network Training

```python
from dezero import Variable, Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L

# Define model
class MLP(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        return self.l2(x)

# Train
model = MLP(100, 10)
optimizer = optimizers.Adam().setup(model)

for epoch in range(100):
    y = model(x)
    loss = F.softmax_cross_entropy(y, t)
    
    model.cleargrads()
    loss.backward()
    optimizer.update()
```

### LSTM Time Series Prediction

```python
from dezero.layers import LSTM, Linear
from dezero.models import Model

class LSTMPredictor(Model):
    def __init__(self, hidden_size):
        super().__init__()
        self.lstm = LSTM(hidden_size)
        self.fc = Linear(1)
    
    def reset_state(self):
        self.lstm.reset_state()
    
    def forward(self, x):
        h = self.lstm(x)
        return self.fc(h)
```

### Transformer (Self-Attention)

```python
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = F.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    attn_weights = F.softmax(scores, axis=-1)
    return F.matmul(attn_weights, V), attn_weights

class MultiHeadAttention(Layer):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.W_q = L.Linear(d_model)
        self.W_k = L.Linear(d_model)
        self.W_v = L.Linear(d_model)
        self.W_o = L.Linear(d_model)
    
    def forward(self, x):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        out, _ = scaled_dot_product_attention(Q, K, V)
        return self.W_o(out)
```

### GPU Acceleration (MLX)

```python
from dezero import cuda

# Check GPU availability
print(f"MLX available: {cuda.gpu_enable}")

# Move data to GPU
x_gpu = cuda.as_gpu(x_numpy)

# Move back to CPU
x_cpu = cuda.as_numpy(x_gpu)
```

## 📁 Project Structure

```
dezero/
├── __init__.py          # Package initialization
├── core.py              # Variable, Function, Parameter
├── cuda.py              # MLX/NumPy backend switching
├── functions.py         # Activation, loss functions
├── functions_conv.py    # Convolution operations
├── layers.py            # Linear, Conv2d, LSTM, etc.
├── models.py            # Model, Sequential, MLP, VGG, ResNet
├── optimizers.py        # SGD, Adam, etc.
├── datasets.py          # MNIST, Spiral, etc.
├── dataloaders.py       # DataLoader, SeqDataLoader
├── transforms.py        # Image transforms
└── utils.py             # Utilities
```

## 🔧 Supported Features

### Layers
| Layer | Description |
|-------|-------------|
| `Linear` | Fully connected layer |
| `Conv2d` | 2D convolution |
| `Deconv2d` | 2D transposed convolution |
| `LSTM` | Long Short-Term Memory |
| `RNN` | Recurrent Neural Network |
| `BatchNorm` | Batch Normalization |
| `EmbedID` | Embedding layer |

### Functions
| Function | Description |
|----------|-------------|
| `relu`, `sigmoid`, `tanh` | Activation functions |
| `softmax`, `log_softmax` | Softmax functions |
| `softmax_cross_entropy` | Cross entropy loss |
| `mean_squared_error` | MSE loss |
| `dropout` | Dropout regularization |
| `conv2d`, `pooling` | Convolution operations |

### Optimizers
| Optimizer | Description |
|-----------|-------------|
| `SGD` | Stochastic Gradient Descent |
| `MomentumSGD` | SGD with momentum |
| `AdaGrad` | Adaptive gradient |
| `Adam` | Adaptive moment estimation |

## 🧪 Running Tests

```bash
python test_dezero.py
```

## 📖 References

- [Deep Learning from Scratch 3](https://www.oreilly.co.jp/books/9784873119069/) - Original DeZero
- [MLX Documentation](https://ml-explore.github.io/mlx/) - Apple's ML framework

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original DeZero by [Koki Saitoh](https://github.com/oreilly-japan/deep-learning-from-scratch-3)
- MLX by [Apple](https://github.com/ml-explore/mlx)

---

<a name="한국어"></a>
# 🍎 DeZero-MLX (한국어)

**밑바닥부터 시작하는 딥러닝 프레임워크 - Apple Silicon GPU 지원**

## ✨ 특징

- 🚀 **Apple Silicon GPU 가속** - MLX 백엔드 지원
- 🔄 **자동 미분** - Define-by-run 동적 계산 그래프
- 🧠 **신경망 레이어** - Linear, Conv2d, LSTM, BatchNorm 등
- ⚡ **옵티마이저** - SGD, Momentum, AdaGrad, Adam
- 📊 **내장 데이터셋** - MNIST, Spiral, SinCurve
- 🔧 **NumPy 폴백** - MLX 없이도 모든 플랫폼에서 작동

## 📦 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/dezero-mlx.git
cd dezero-mlx

# 의존성 설치
pip install numpy

# (선택) Apple Silicon GPU 가속을 위한 MLX 설치
pip install mlx
```

## 🚀 빠른 시작

```python
import numpy as np
from dezero import Variable

# 자동 미분
x = Variable(np.array(2.0))
y = x ** 2 + 3 * x + 1
y.backward()

print(f"y = {y.data}")       # y = 11.0
print(f"dy/dx = {x.grad}")   # dy/dx = 7.0
```

## 📚 예제

자세한 예제는 다음 파일들을 참고하세요:
- `test_dezero_full.py` - 기본 기능 테스트
- `test_advanced_models.py` - LSTM, Transformer 테스트

## 📖 참고자료

- [밑바닥부터 시작하는 딥러닝 3](https://www.hanbit.co.kr/store/books/look.php?p_code=B6627606922) - 원본 DeZero
- [MLX 문서](https://ml-explore.github.io/mlx/) - Apple ML 프레임워크
