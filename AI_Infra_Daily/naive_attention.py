import torch
import torch.nn.functional as F


class NaiveAttention:
    """Scaled dot-product attention (no learnable parameters)."""

    def __call__(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        **B** = batch, **H** = num_heads, **N** = seq_len, **d** = head_dim
        Args:
            Q: (B, H, N, d)
            K: (B, H, N, d)
            V: (B, H, N, d)
        Returns:
            output: (B, H, N, d)
        """
        d_k = Q.size(-1)
        KT = K.transpose(-2,-1)
        QK = Q.matmul(KT) #B, H, N, N
        QK = QK/(d_k ** 0.5)
        attn_wrights = QK.softmax(dim = -1)  #B, H, N, N
        output = attn_wrights.matmul(V)

        return output


def test_output_shape():
    attn = NaiveAttention()
    B, H, N, d = 2, 4, 8, 64
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)
    out = attn(Q, K, V)
    assert out.shape == (B, H, N, d), f"Expected {(B, H, N, d)}, got {out.shape}"


def test_single_token():
    """With seq_len=1, output must equal V (only one key to attend to)."""
    attn = NaiveAttention()
    B, H, N, d = 1, 1, 1, 4
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)
    out = attn(Q, K, V)
    assert torch.allclose(out, V), "Single token output should equal V"


def test_attn_weights_sum_to_one():
    """Softmax rows should sum to 1."""
    attn = NaiveAttention()
    B, H, N, d = 1, 1, 4, 8
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums)), "Each row must sum to 1"


def test_identical_keys_uniform_attention():
    """When all keys are identical, attention weights should be uniform."""
    attn = NaiveAttention()
    B, H, N, d = 1, 1, 4, 8
    Q = torch.randn(B, H, N, d)
    k = torch.randn(1, 1, 1, d)
    K = k.expand(B, H, N, d)
    V = torch.randn(B, H, N, d)
    out = attn(Q, K, V)
    expected = V.mean(dim=2, keepdim=True).expand_as(V)
    assert torch.allclose(out, expected, atol=1e-6), "Uniform attention should produce mean of V"


def test_matches_pytorch_sdpa():
    """Output should match PyTorch's built-in scaled_dot_product_attention."""
    attn = NaiveAttention()
    B, H, N, d = 2, 4, 16, 32
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)
    out = attn(Q, K, V)
    ref = F.scaled_dot_product_attention(Q, K, V)
    assert torch.allclose(out, ref, atol=1e-5), "Should match PyTorch SDPA"


if __name__ == "__main__":
    test_output_shape()
    test_single_token()
    test_attn_weights_sum_to_one()
    test_identical_keys_uniform_attention()
    test_matches_pytorch_sdpa()
    print("All tests passed!")
