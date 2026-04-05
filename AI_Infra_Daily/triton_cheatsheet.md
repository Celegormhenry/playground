# Triton Kernel Cheatsheet

## Program ID & Grid

```python
# kernel 里获取当前 program 在 grid 中的位置
pid_x = tl.program_id(0)   # grid axis 0
pid_y = tl.program_id(1)   # grid axis 1

# 整除向上取整 (kernel 内和 Python 侧都能用)
tl.cdiv(N, BLOCK_N)         # = ceil(N / BLOCK_N)
triton.cdiv(N, BLOCK_N)     # Python 侧同名函数

# 从 flat pid 解码出 2D tile index（matmul 常用模式）
num_col_tiles = tl.cdiv(N, BLOCK_N)
pid_row = pid // num_col_tiles
pid_col = pid  % num_col_tiles
```

---

## Offset & Mask

```python
# 创建连续 offset 数组
offs = tl.arange(0, BLOCK)       # → [0, 1, 2, ..., BLOCK-1]

# 平移到当前 tile 的起始位置
row_offs = pid_row * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
col_offs = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)   # (BLOCK_N,)

# 边界 mask（防止读写越界）
mask = row_offs < M              # 1D mask
mask_2d = row_offs[:, None] < M  # broadcast 成 2D
```

---

## 指针算术 (2D Indexing)

```python
# 核心模式：base + row * stride_row + col * stride_col
# [:, None] 和 [None, :] 构成 2D 网格

# 例: 加载 (BLOCK_M, D) 的 tile
ptrs = (base_ptr
        + batch_head_id * stride_bh           # 选 batch*head
        + row_offs[:, None] * stride_row       # 行方向
        + col_offs[None, :] * stride_col)      # 列方向

# Python 侧获取 stride
tensor.stride(0)    # dim 0 每跨一步要跳几个 element
tensor.stride(1)    # dim 1
tensor.stride(2)    # dim 2
```

---

## Load & Store

```python
# 加载数据到 SRAM
data = tl.load(ptrs, mask=mask, other=0.0)
#   mask=False 的位置填 other 值
#   other=float('-inf') 常用于 softmax 的无效位置

# 写回 HBM
tl.store(ptrs, data, mask=mask)
```

---

## 矩阵乘法 & 转置

```python
# C = A @ B，A 和 B 必须是 2D，dtype 要一致
C = tl.dot(A, B, allow_tf32=False)
#   allow_tf32=False → 保持 fp32 精度（TF32 会截断 mantissa）

# 转置
K_T = tl.trans(K_tile)        # (BLOCK_N, D) → (D, BLOCK_N)

# 常见组合: S = Q @ K^T
S = tl.dot(Q_tile, tl.trans(K_tile), allow_tf32=False)

# dot 要求两个输入 dtype 一致，不一致时手动转换
tl.dot(P.to(V_tile.dtype), V_tile, allow_tf32=False)
```

---

## Reduce 操作

```python
# 沿某个 axis 做 reduce
row_max = tl.max(S, axis=1)     # (M, N) → (M,)  每行最大值
row_sum = tl.sum(P, axis=1)     # (M, N) → (M,)  每行求和

col_max = tl.max(S, axis=0)     # (M, N) → (N,)  每列最大值
```

---

## 数学运算

```python
tl.exp(x)              # element-wise e^x
tl.log(x)              # element-wise ln(x)
tl.maximum(a, b)       # element-wise max（两个 tensor 逐元素取大）
tl.abs(x)              # element-wise 绝对值
tl.sqrt(x)             # element-wise 平方根
```

---

## 初始化

```python
# 全零
acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

# 全填某个值
m = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
ones = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
```

---

## 条件选择

```python
# where: 类似 numpy.where
result = tl.where(condition, value_true, value_false)

# 典型用法: mask 掉 out-of-bounds 的 key，让 softmax 忽略它们
S = tl.where(kv_mask[None, :], S, float('-inf'))
#   valid 位置保留 S，invalid 位置变 -inf → exp(-inf) = 0
```

---

## Broadcasting 规则

```python
# Triton 的 broadcasting 和 NumPy 一样
# (BLOCK_M,) 和 (BLOCK_M, D) 运算时需要手动扩维

correction = tl.exp(m - m_new)              # (BLOCK_M,)
O_acc = O_acc * correction[:, None]         # (BLOCK_M, D) * (BLOCK_M, 1)
P = tl.exp(S - m_new[:, None])             # (BLOCK_M, N) - (BLOCK_M, 1)
O_final = O_acc / l[:, None]               # (BLOCK_M, D) / (BLOCK_M, 1)
```

---

## Tensor 预处理 (Python 侧)

```python
# Flatten batch & head → 一个维度，简化 kernel 索引
# (B, H, N, d) → (B*H, N, d)
Q_flat = Q.reshape(B * H, N, d).contiguous()
#   .contiguous() 确保内存连续，stride 才正确

# 还原
O = O_flat.reshape(B, H, N, d)

# 分配输出（和输入同 shape/device/dtype）
O = torch.empty_like(Q_flat)
# 或指定 shape/dtype
S = torch.empty(B * H, N, N, device=Q.device, dtype=torch.float32)
```

---

## Python 侧启动 Kernel

```python
# 定义 grid（有几个 program 实例）
grid = (triton.cdiv(N, BLOCK_M), B * H)

# 启动 kernel（方括号里放 grid）
my_kernel[grid](
    # 指针: tensor 直接传，Triton 自动取 .data_ptr()
    Q_flat, K_flat, V_flat, O,
    # 标量参数
    N,
    # Strides
    Q_flat.stride(0), Q_flat.stride(1), Q_flat.stride(2),
    K_flat.stride(0), K_flat.stride(1), K_flat.stride(2),
    # Scale
    scale,
    # Compile-time constants (constexpr)
    BLOCK_M=64, BLOCK_N=64, D=d,
    # Pipeline stages
    num_stages=1,
)
```

---

## Flash Attention 专用模式

```python
# ── Online Softmax 单次迭代 ──
# 处理一个 KV tile 后更新 running statistics:

m_block = tl.max(S, axis=1)                    # 当前 block 的行最大值
m_new = tl.maximum(m, m_block)                  # 和历史 max 比较
correction = tl.exp(m - m_new)                  # 旧统计量的修正因子
P = tl.exp(S - m_new[:, None])                  # 当前 block 的 exp scores
l = l * correction + tl.sum(P, axis=1)          # 更新 running sum
O_acc = O_acc * correction[:, None] + tl.dot(P.to(V_tile.dtype), V_tile, allow_tf32=False)
m = m_new                                        # 更新 running max

# ── 循环结束后 deferred normalization ──
O_acc = O_acc / l[:, None]
```
