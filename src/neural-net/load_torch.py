import torch





data = torch.load("../../data/processed/neural-stuff/encodings.pt", map_location="cpu")
for i, t in enumerate( data["test"]):
    print(i, t)
