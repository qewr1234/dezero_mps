#!/usr/bin/env python3
"""
DeZero 고급 모델 테스트
- LSTM: 시계열 예측 (Sin 곡선)
- Transformer: Self-Attention 기반 분류
"""
import sys
sys.path.append('/Users/dbwns/프로그래밍/DeepLearning/dezero-mlx')

import numpy as np
import dezero
from dezero import Variable, Parameter, Config
from dezero import functions as F
from dezero import layers as L
from dezero.layers import Layer
from dezero.models import Model
from dezero import optimizers
from dezero import cuda

print("=" * 70)
print("🧠 DeZero 고급 모델 테스트 (LSTM & Transformer)")
print("=" * 70)
print(f"GPU(MLX) 사용 가능: {cuda.gpu_enable}")


# =============================================================================
# Part 1: LSTM - 시계열 예측 (Sin 곡선)
# =============================================================================
print("\n" + "=" * 70)
print("📈 Part 1: LSTM 시계열 예측 (Sin 곡선)")
print("=" * 70)


class LSTMModel(Model):
    """LSTM 기반 시계열 예측 모델"""
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.lstm = L.LSTM(hidden_size)
        self.fc = L.Linear(out_size)
    
    def reset_state(self):
        self.lstm.reset_state()
    
    def forward(self, x):
        h = self.lstm(x)
        y = self.fc(h)
        return y


def generate_sin_data(seq_len=1000, train_ratio=0.8):
    """Sin 곡선 데이터 생성"""
    t = np.linspace(0, 4 * np.pi, seq_len)
    y = np.sin(t).astype(np.float32)
    
    X = y[:-1].reshape(-1, 1)
    T = y[1:].reshape(-1, 1)
    
    split = int(len(X) * train_ratio)
    return (X[:split], T[:split]), (X[split:], T[split:])


# 데이터 준비
(X_train, T_train), (X_test, T_test) = generate_sin_data()
print(f"\n학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")

# 모델 & 옵티마이저
hidden_size = 32
model = LSTMModel(hidden_size, out_size=1)
optimizer = optimizers.Adam(alpha=0.001).setup(model)

# 학습 파라미터
seq_length = 30
epochs = 100

print(f"\nLSTM 학습 시작 (hidden: {hidden_size}, seq_len: {seq_length})")
print("-" * 50)

for epoch in range(epochs):
    model.reset_state()
    total_loss = 0
    loss_count = 0
    
    for i in range(0, len(X_train) - seq_length, seq_length):
        x_batch = X_train[i:i+seq_length]
        t_batch = T_train[i:i+seq_length]
        
        loss = Variable(np.array(0.0, dtype=np.float32))
        for t in range(seq_length):
            x = Variable(x_batch[t:t+1])
            target = Variable(t_batch[t:t+1])
            y = model(x)
            loss = loss + F.mean_squared_error(y, target)
        
        loss = loss / seq_length
        
        model.cleargrads()
        loss.backward()
        optimizer.update()
        
        total_loss += float(loss.data)
        loss_count += 1
        
        model.lstm.reset_state()
    
    avg_loss = total_loss / loss_count if loss_count > 0 else 0
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.6f}")

# 테스트
print("-" * 50)
print("테스트 중...")

model.reset_state()
predictions = []
actuals = []

with dezero.no_grad():
    for i in range(len(X_test)):
        x = Variable(X_test[i:i+1])
        y = model(x)
        predictions.append(float(y.data.flatten()[0]))
        actuals.append(float(T_test[i].flatten()[0]))

mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
print(f"테스트 MSE: {mse:.6f}")
print(f"예측 샘플: {predictions[:5]}")
print(f"실제 샘플: {actuals[:5]}")


# =============================================================================
# Part 2: Transformer - Self-Attention 기반 분류
# =============================================================================
print("\n" + "=" * 70)
print("🤖 Part 2: Transformer (Self-Attention) 분류")
print("=" * 70)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled Dot-Product Attention
    
    Args:
        Q: (batch, seq_len, d_k)
        K: (batch, seq_len, d_k)
        V: (batch, seq_len, d_v)
        mask: optional mask
    
    Returns:
        output: (batch, seq_len, d_v)
        attn_weights: (batch, seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    
    # K^T: (batch, seq, d_k) -> (batch, d_k, seq)
    K_T = K.transpose(0, 2, 1)
    
    # Attention scores: (batch, seq, seq)
    # 사용: F.batch_matmul (3D 텐서용)
    scores = F.batch_matmul(Q, K_T) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores + mask * (-1e9)
    
    # Softmax over last axis
    attn_weights = F.softmax(scores, axis=-1)
    
    # Weighted sum: (batch, seq, seq) @ (batch, seq, d_v) -> (batch, seq, d_v)
    output = F.batch_matmul(attn_weights, V)
    
    return output, attn_weights


class MultiHeadAttention(Layer):
    """Multi-Head Attention"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = L.Linear(d_model, in_size=d_model)
        self.W_k = L.Linear(d_model, in_size=d_model)
        self.W_v = L.Linear(d_model, in_size=d_model)
        self.W_o = L.Linear(d_model, in_size=d_model)
    
    def forward(self, x):
        """Self-Attention: Q=K=V=x"""
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        output, _ = scaled_dot_product_attention(Q, K, V)
        output = self.W_o(output)
        return output


class PositionwiseFeedForward(Layer):
    """Position-wise Feed-Forward Network"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = L.Linear(d_ff, in_size=d_model)
        self.fc2 = L.Linear(d_model, in_size=d_ff)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TransformerEncoderLayer(Layer):
    """Single Transformer Encoder Layer"""
    def __init__(self, d_model, n_heads, d_ff, dropout_ratio=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.dropout_ratio = dropout_ratio
    
    def forward(self, x):
        # Self-Attention + Residual
        attn_out = self.self_attn(x)
        attn_out = F.dropout(attn_out, self.dropout_ratio)
        x = x + attn_out
        
        # FFN + Residual
        ffn_out = self.ffn(x)
        ffn_out = F.dropout(ffn_out, self.dropout_ratio)
        x = x + ffn_out
        
        return x


class PositionalEncoding(Layer):
    """Positional Encoding"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe[np.newaxis, :, :]
    
    def forward(self, x):
        seq_len = x.shape[1]
        pe = Variable(self.pe[:, :seq_len, :].astype(np.float32))
        return x + pe


class TransformerClassifier(Model):
    """Transformer 기반 분류 모델"""
    def __init__(self, input_dim, d_model, n_heads, d_ff, n_layers, n_classes):
        super().__init__()
        
        self.input_proj = L.Linear(d_model, in_size=input_dim)
        self.pos_encoding = PositionalEncoding(d_model)
        
        self.encoder_layers = []
        for i in range(n_layers):
            layer = TransformerEncoderLayer(d_model, n_heads, d_ff)
            setattr(self, f'encoder_{i}', layer)
            self.encoder_layers.append(layer)
        
        self.classifier = L.Linear(n_classes, in_size=d_model)
    
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Global average pooling
        seq_len = x.shape[1]
        x = F.sum(x, axis=1) / seq_len
        
        # Classification
        logits = self.classifier(x)
        return logits


# Spiral 데이터 준비
print("\n데이터 준비 중...")

from dezero.datasets import Spiral
train_data = Spiral(train=True)
test_data = Spiral(train=False)


def augment_to_sequence(data, seq_len=8):
    """각 2D 포인트를 seq_len 길이의 시퀀스로 확장"""
    n_samples = len(data)
    augmented = np.zeros((n_samples, seq_len, 2), dtype=np.float32)
    
    for i in range(n_samples):
        point = data[i]
        for j in range(seq_len):
            noise = np.random.randn(2).astype(np.float32) * 0.1
            augmented[i, j] = point + noise * (j / seq_len)
    
    return augmented


seq_len = 8
X_train = augment_to_sequence(train_data.data, seq_len)
T_train = train_data.label
X_test = augment_to_sequence(test_data.data, seq_len)
T_test = test_data.label

print(f"학습 데이터: {X_train.shape}")
print(f"테스트 데이터: {X_test.shape}")

# 모델 설정
d_model = 32
n_heads = 4
d_ff = 64
n_layers = 2
n_classes = 3

model = TransformerClassifier(
    input_dim=2,
    d_model=d_model,
    n_heads=n_heads,
    d_ff=d_ff,
    n_layers=n_layers,
    n_classes=n_classes
)
optimizer = optimizers.Adam(alpha=0.001).setup(model)

print(f"\nTransformer 설정:")
print(f"  d_model: {d_model}, n_heads: {n_heads}")
print(f"  d_ff: {d_ff}, n_layers: {n_layers}")
print(f"  seq_len: {seq_len}")

# 학습
epochs = 100
batch_size = 32

print(f"\nTransformer 학습 시작")
print("-" * 50)

for epoch in range(epochs):
    perm = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[perm]
    T_train_shuffled = T_train[perm]
    
    total_loss = 0
    total_acc = 0
    n_batches = 0
    
    for i in range(0, len(X_train), batch_size):
        x = Variable(X_train_shuffled[i:i+batch_size])
        t = Variable(T_train_shuffled[i:i+batch_size])
        
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        
        model.cleargrads()
        loss.backward()
        optimizer.update()
        
        total_loss += float(loss.data)
        total_acc += float(acc.data)
        n_batches += 1
    
    if (epoch + 1) % 20 == 0:
        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches
        print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

# 테스트
print("-" * 50)
print("테스트 중...")

with dezero.no_grad():
    x = Variable(X_test)
    t = Variable(T_test)
    y = model(x)
    test_acc = F.accuracy(y, t)
    print(f"테스트 정확도: {float(test_acc.data):.4f}")


# =============================================================================
# Part 3: Attention 시각화
# =============================================================================
print("\n" + "=" * 70)
print("👁️ Part 3: Self-Attention 작동 확인")
print("=" * 70)

print("\nSelf-Attention 연산 테스트:")

batch_size = 2
seq_len = 4
d_model = 8

x = Variable(np.random.randn(batch_size, seq_len, d_model).astype(np.float32))
print(f"입력 shape: {x.shape}")

output, weights = scaled_dot_product_attention(x, x, x)

print(f"출력 shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"\nAttention weights (첫 번째 배치):")
print(weights.data[0])

print("\n✓ Attention weights 합계 (각 행이 1이어야 함):")
print(weights.data[0].sum(axis=-1))


# =============================================================================
# Part 4: 간단한 언어 모델 (Character-level)
# =============================================================================
print("\n" + "=" * 70)
print("📝 Part 4: Character-level 언어 모델 (LSTM)")
print("=" * 70)


class CharLSTM(Model):
    """Character-level LSTM 언어 모델"""
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = L.EmbedID(vocab_size, embed_size)
        self.lstm = L.LSTM(hidden_size, in_size=embed_size)
        self.fc = L.Linear(vocab_size, in_size=hidden_size)
    
    def reset_state(self):
        self.lstm.reset_state()
    
    def forward(self, x):
        h = self.embed(x)
        h = self.lstm(h)
        y = self.fc(h)
        return y


# 간단한 텍스트 데이터
text = "hello world. deep learning is amazing. neural networks are powerful."
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

print(f"\n텍스트 길이: {len(text)}")
print(f"고유 문자 수: {vocab_size}")
print(f"문자: {chars}")

# 데이터 준비
X_text = np.array([char_to_idx[ch] for ch in text[:-1]], dtype=np.int32)
T_text = np.array([char_to_idx[ch] for ch in text[1:]], dtype=np.int32)

# 모델
embed_size = 16
hidden_size = 32
model = CharLSTM(vocab_size, embed_size, hidden_size)
optimizer = optimizers.Adam(alpha=0.01).setup(model)

# 학습
epochs = 200
seq_length = 20

print(f"\nChar-LSTM 학습 시작")
print("-" * 50)

for epoch in range(epochs):
    model.reset_state()
    total_loss = 0
    loss_count = 0
    
    for i in range(0, len(X_text) - seq_length, seq_length):
        x_seq = X_text[i:i+seq_length]
        t_seq = T_text[i:i+seq_length]
        
        loss = Variable(np.array(0.0, dtype=np.float32))
        for j in range(seq_length):
            x = Variable(np.array([x_seq[j]], dtype=np.int32))
            t = Variable(np.array([t_seq[j]], dtype=np.int32))
            y = model(x)
            loss = loss + F.softmax_cross_entropy(y, t)
        
        loss = loss / seq_length
        
        model.cleargrads()
        loss.backward()
        optimizer.update()
        
        total_loss += float(loss.data)
        loss_count += 1
        
        model.reset_state()
    
    if (epoch + 1) % 50 == 0:
        avg_loss = total_loss / loss_count
        print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f}")

# 텍스트 생성
print("-" * 50)
print("텍스트 생성 테스트:")


def generate_text(model, start_char, length=50):
    model.reset_state()
    result = start_char
    
    with dezero.no_grad():
        x = Variable(np.array([char_to_idx[start_char]], dtype=np.int32))
        
        for _ in range(length):
            y = model(x)
            probs = F.softmax(y, axis=1).data.flatten()
            
            # numpy로 변환 (MLX 호환)
            if hasattr(probs, 'tolist'):
                probs = np.array(probs)
            
            # 확률 정규화
            probs = np.clip(probs, 1e-10, 1.0)
            probs = probs / probs.sum()
            
            next_idx = np.random.choice(len(probs), p=probs)
            next_char = idx_to_char[next_idx]
            result += next_char
            x = Variable(np.array([next_idx], dtype=np.int32))
    
    return result


generated = generate_text(model, 'h', length=50)
print(f"시작 문자 'h'로 생성: {generated}")

generated = generate_text(model, 'd', length=50)
print(f"시작 문자 'd'로 생성: {generated}")


# =============================================================================
# 완료
# =============================================================================
print("\n" + "=" * 70)
print("✅ 고급 모델 테스트 완료!")
print("=" * 70)
print("""
📌 구현된 기능:
  • LSTM: 시계열 예측 (Sin 곡선)
  • Transformer: Self-Attention, Multi-Head Attention, FFN
  • Positional Encoding
  • Character-level 언어 모델
  • batch_matmul: 3D 텐서 배치 행렬곱
  • dropout: 임의 shape 텐서 지원
  
💡 추가 확장 아이디어:
  • Layer Normalization 구현
  • Decoder (Cross-Attention) 구현  
  • Seq2Seq 번역 모델
  • GPT 스타일 언어 모델
""")
