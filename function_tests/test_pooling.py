import torch
import sys
import os

# make sure the project root is on the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.pooling import build_pooling, build_unpooling


# --- Config (mirrors the real config structure) ---
model_paras = {
    "out_size_ope": 8,
    "pooling": {
        "method": "topk",
        "ratio":  0.5,
        "num_layers": 1,
    }
}

# --- Dummy data ---
B  = 4    # batch size
N  = 20   # max number of operations (padded)
d  = model_paras["out_size_ope"]

torch.manual_seed(42)
h         = torch.randn(B, N, d)
adj       = torch.randint(0, 2, (B, N, N)).float()
nums_opes = torch.tensor([20, 18, 15, 20])  # instance 2 has only 15 real ops

pool   = build_pooling(model_paras)
unpool = build_unpooling(model_paras)

print("=" * 50)
print("TEST 1: output shapes")
print("=" * 50)
h_pooled, adj_pooled, pool_info = pool(h, adj, nums_opes)
# k = max(1, int(0.5 * 15)) = 7  (conservative: min of nums_opes)
k_expected = max(1, int(model_paras["pooling"]["ratio"] * nums_opes.min().item()))
assert h_pooled.shape   == (B, k_expected, d),  f"h_pooled shape wrong: {h_pooled.shape}"
assert adj_pooled.shape == (B, k_expected, k_expected), f"adj_pooled shape wrong: {adj_pooled.shape}"
print(f"  h_pooled:   {h_pooled.shape}   expected ({B}, {k_expected}, {d})")
print(f"  adj_pooled: {adj_pooled.shape}   expected ({B}, {k_expected}, {k_expected})")
print("  PASSED")

print()
print("=" * 50)
print("TEST 2: no padding node is ever selected")
print("=" * 50)
top_idx = pool_info["top_idx"]   # [B, k]
for b in range(B):
    max_idx = top_idx[b].max().item()
    limit   = nums_opes[b].item()
    assert max_idx < limit, f"batch {b}: selected index {max_idx} >= nums_opes {limit}"
    print(f"  batch {b}: max selected index = {max_idx}, nums_opes = {limit}   PASSED")

print()
print("=" * 50)
print("TEST 3: unpool reconstructs original size")
print("=" * 50)
h_unpooled, adj_out = unpool(h_pooled, pool_info, h, adj)
assert h_unpooled.shape == (B, N, d), f"h_unpooled shape wrong: {h_unpooled.shape}"
assert adj_out.shape    == (B, N, N), f"adj_out shape wrong: {adj_out.shape}"
print(f"  h_unpooled: {h_unpooled.shape}   expected ({B}, {N}, {d})")
print("  PASSED")

print()
print("=" * 50)
print("TEST 4: skip connection is added correctly")
print("=" * 50)
# at a selected position, h_unpooled must differ from h_pooled
# because skip_h (which equals h here) was added
for b in range(B):
    idx = top_idx[b, 0].item()
    val_unpooled = h_unpooled[b, idx]
    val_pooled   = h_pooled[b, 0]
    assert not torch.allclose(val_unpooled, val_pooled), \
        f"batch {b}: skip connection seems to have no effect"
print("  PASSED")

print()
print("=" * 50)
print("TEST 5: non-selected positions are only skip_h (scatter fills zeros)")
print("=" * 50)
# at a non-selected position, h_unpooled must equal skip_h exactly
# because scatter wrote 0 there and then skip_h was added
for b in range(B):
    selected = set(top_idx[b].tolist())
    all_positions = set(range(nums_opes[b].item()))
    non_selected = list(all_positions - selected)
    if len(non_selected) > 0:
        pos = non_selected[0]
        assert torch.allclose(h_unpooled[b, pos], h[b, pos]), \
            f"batch {b}, position {pos}: expected skip_h value"
print("  PASSED")

print()
print("=" * 50)
print("TEST 6: gradient flows back to p")
print("=" * 50)
# fresh forward pass with gradient tracking
h_grad = torch.randn(B, N, d, requires_grad=True)
h_p, _, _ = pool(h_grad, adj, nums_opes)
loss = h_p.sum()
loss.backward()
assert pool.proj.weight.grad is not None,             "proj.weight.grad is None"
assert not torch.all(pool.proj.weight.grad == 0),     "proj.weight.grad is all zeros"
print(f"  proj.weight.grad norm: {pool.proj.weight.grad.norm():.6f}")
print("  PASSED")

print()
print("=" * 50)
print("TEST 7: adj values are binary after graph power")
print("=" * 50)
unique_vals = adj_pooled.unique()
assert set(unique_vals.tolist()).issubset({0.0, 1.0}), \
    f"adj_pooled contains non-binary values: {unique_vals}"
print(f"  unique values in adj_pooled: {unique_vals.tolist()}")
print("  PASSED")

print()
print("=" * 50)
print("TEST 8: two pool steps, graph shrinks each time")
print("=" * 50)
h1, adj1, info1 = pool(h, adj, nums_opes)
k1 = h1.shape[1]
nums_opes_1 = torch.full((B,), k1, dtype=torch.long)
h2, adj2, info2 = pool(h1, adj1, nums_opes_1)
k2 = h2.shape[1]
assert k2 < k1, f"second pool did not reduce size: k1={k1}, k2={k2}"
print(f"  after pool 1: {k1} nodes")
print(f"  after pool 2: {k2} nodes")
print("  PASSED")

print()
print("=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)
