import torch
import argparse
import torch.nn.functional as F




def load_encoded_sentences(tensor_path, split="train"):
    enc = torch.load(tensor_path, map_location="cpu") 
    X = enc[split]             
    y = enc[f"y_{split}"]       
    label2id = enc.get("label2id", {})


    return X, y, label2id



def make_random_weights( input_dim, hidden_dim, num_classes,  seed=0):
    torch.manual_seed(seed)
    W1 = torch.randn(hidden_dim, input_dim+1)
    W2 = torch.randn(num_classes, hidden_dim)

    with torch.no_grad():
        W1.mul_(0.01)
        W2.mul_(0.01)
    W1.requires_grad_()
    W2.requires_grad_()
    return {"W1": W1, "W2": W2}


def forward(X, params: dict):
    '''
    Added the bias as 1s 
    '''
    bias = torch.ones(X.size(0), 1)
    X = torch.cat((X, bias), dim=1)

    W1 = params["W1"]; W2 = params["W2"]

    z1 = X @ W1.t()
    h1 = torch.tanh(z1)
    logits = h1 @ W2.t()
    return logits, h1
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", "--hidden_dim", type=int, default=512)
    parser.add_argument("-l", "--learning_rate", default=0.1)
    parser.add_argument("-e", "--epochs", default=500)
    args = parser.parse_args()

    X, y, label2id = load_encoded_sentences("./encodings.pt")
    N, input_dim = X.shape
    num_classes = int(y.max().item() +1)
    hidden_dim = args.hidden_dim 
    params = make_random_weights(input_dim, hidden_dim, num_classes)

    epochs = args.epochs
    lr = args.learning_rate

    # Gradient Descent
    print("epoch  Loss    ret")
    for epoch in range(1, int(epochs)+1):
        logits, _ = forward(X, params)
        loss = F.cross_entropy(logits, y)

        loss.backward()

        with torch.no_grad():
            for k in params:
                params[k] -= lr * params[k].grad
                params[k].grad.zero_()

        if epoch % 10 ==0:
            ret = (logits.argmax(dim=1) ==y).float().mean().item()
            print(epoch, loss.item(), ret)


    ckpt_path = "./simple_model.pt"
    torch.save({
        "params": {k: v.detach().cpu() for k, v in params.items()},  
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
        "uses_bias_column": True, 
        "label2id": label2id,
    }, ckpt_path)
    print("Saved model to:", ckpt_path)
    



