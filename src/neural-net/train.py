import torch
import argparse
import torch.nn.functional as F




def load_encoded_sentences(tensor_path, label_path=None):
    encodings = torch.load(tensor_path)
    #labels = torch.load(label_path)
    X = encodings["train"]
    #y = labels["train"]
    return X #, y


def make_random_weights( input_dim, hidden_dim, num_classes,  seed=0):
    torch.manual_seed(seed)
    W1 = torch.randn(hidden_dim, input_dim, requires_grad=True) * 0.01
    W2 = torch.randn(num_classes, hidden_dim, requires_grad=True) *0.01
    return {"W1": W1, "W2": W2}


def forward(X, params: dict):
    '''
    Bias just added as an extra column of 1s
    '''
    bias = torch.ones(X.size(0), 1)
    X = torch.cat((X, bias), dim=1)

    W1 = params["W1"]; W2 = params["W2"]

    # First hidden layer
    z1 = X @ W1.t()
    h1 = torch.tanh(z1)
    logits = h1 @ W2.t()
    return logits, h1
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-h", "--hidden_dim")
    parser.add_argument("-l", "--learning_rate", default=0.1)
    parser.add_argument("-e", "--epochs", default=100)
    args = parser.parse_args()

    X, y = load_encoded_sentences("../../data/processed/neural-stuff/encodings.pt")
    N, input_dim = X.shape
    num_classes = int(y.max().item() +1)
    hidden_dim = args.hidden_dim 
    params = make_random_weights(input_dim, hidden_dim, num_classes)

    epochs = args.epochs
    lr = args.learning_rate

    # Gradient Descent
    for epoch in range(1, epochs+1):
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


        



