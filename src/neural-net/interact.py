import torch
import numpy as np
from gensim.models import KeyedVectors

ENC_PATH   = "./encodings.pt"
MODEL_PATH = "./simple_model.pt"
EMB_PATH   = "./glove_embeddings.data"

def sentence_vector(sentence, kv):
    tokens = sentence.split()
    vecs = [kv[w] for w in tokens if w in kv.key_to_index]
    

    if not vecs:
        return np.zeros(kv.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)





def forward(X, params):
    b  = torch.ones(X.size(0), 1, dtype=X.dtype)
    Xb = torch.cat([X, b], dim=1)           
    h  = torch.tanh(Xb @ params["W1"].t())


    return h @ params["W2"].t()           

if __name__ == "__main__":
    enc = torch.load(ENC_PATH, map_location="cpu")
    id2label = {v: k for k, v in enc["label2id"].items()}
    names = [id2label[i] for i in range(len(id2label))]

    ck = torch.load(MODEL_PATH, map_location="cpu")
    params = {k: v.float() for k, v in ck["params"].items()}

    kv = KeyedVectors.load(EMB_PATH)

    print("Type text and press Enter(type 'quit' to exit).")
    print("Classes:", ", ".join(names))
    while True:
        s = input("> ").strip()
        if not s:
            continue
        if s.lower() in {"quit", "exit"}:
            break

        vec = sentence_vector(s, kv)
        X = torch.from_numpy(vec).unsqueeze(0).float()

        with torch.no_grad():
            logits = forward(X, params)
            probs = torch.softmax(logits, dim=1).squeeze(0)  

        order = torch.argsort(probs, descending=True).tolist()
        for idx in order:
            print(f"{names[idx]:16s}  {probs[idx].item():.3f}")

