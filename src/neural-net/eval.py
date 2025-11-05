import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score

ENC_PATH   = "../../data/processed/neural-stuff/encodings.pt"
MODEL_PATH = "../../data/processed/neural-stuff/simple_model.pt"

def forward(X, params):
    b  = torch.ones(X.size(0), 1, dtype=X.dtype)
    Xb = torch.cat([X, b], dim=1)           
    h  = torch.tanh(Xb @ params["W1"].t())
    return h @ params["W2"].t()          

if __name__ == "__main__":


    enc = torch.load(ENC_PATH, map_location="cpu")
    X   = enc["test"].float()
    y   = enc["y_test"].long()
    id2label = {v: k for k, v in enc["label2id"].items()}
    names = [id2label[i] for i in range(len(id2label))]

    ck = torch.load(MODEL_PATH, map_location="cpu")
    params = {k: v.float() for k, v in ck["params"].items()}

    with torch.no_grad():
        logits = forward(X, params)
        yhat   = logits.argmax(1)
        loss   = F.cross_entropy(logits, y).item()

    print(f"TEST loss: {loss:.6f}")
    print(f"TEST acc : {accuracy_score(y.numpy(), yhat.numpy()):.6f}")
    print(classification_report(y.numpy(), yhat.numpy(), target_names=names, digits=3))

