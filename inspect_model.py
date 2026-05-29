import torch

checkpoint = torch.load("./results/save_10_5.pt", map_location="cpu")

print(f"Typ: {type(checkpoint)}")
print(f"Anzahl Layer: {len(checkpoint)}\n")

total_params = 0
for name, tensor in checkpoint.items():
    params = tensor.numel()
    total_params += params
    print(f"{name}")
    print(f"  Shape: {list(tensor.shape)}  |  Params: {params}  |  dtype: {tensor.dtype}")
    print(f"  min={tensor.min():.6f}  max={tensor.max():.6f}  mean={tensor.mean():.6f}\n")

print(f"Gesamt Parameter: {total_params:,}")
