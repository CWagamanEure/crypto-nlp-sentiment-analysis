import torch
import argparse
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

def build_dataset(token_ids, block_size):
    num_tokens = token_ids.size(0)

    # Dont wanna have to pad, so reducing to tokens minus size so
    # that all windows are same size
    max_start = num_tokens - (block_size+1)
    if max_start <0:
        raise ValueError("Not enough tokens for block_size+1")

    starts = list(range(0, max_start+1))
    n_examples = len(starts)

    inputs = torch.zeros((n_examples, block_size), dtype=torch.long)
    targets = torch.zeros((n_examples, block_size), dtype=torch.long)

    for i, start in enumerate(starts):
        inputs[i] = token_ids[start : start + block_size]
        targets[i] = token_ids[start+1: start + block_size+1]

    return TensorDataset(inputs, targets)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path")
    parser.add_argument("-b", "--block_size", default=8)
    parser.add_argument("-ba", "--batch_size", default=128)
    parser.add_argument("-lr", "--learning_rate", default=5e-5)
    parser.add_argument("-e", "--epochs", default=10)
    parser.add_argument("-o", "--output_dir", default="../../models/")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    with open(args.path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded corpus from {args.path}")

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_ids = torch.tensor(token_ids, dtype=torch.long)
    print("Beep boop beep... Tokenizing")

    dataset = build_dataset(token_ids, args.block_size)
    print(f"You have {len(dataset)} training samples")
    if len(dataset) > 1000000: print("Holy cow thats alot")
    else: print("Should take no time at all")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    for epoch in range(1, args.epochs +1):
        total_loss = 0
        num_batches = 0

        for batch_index, batch in enumerate(dataloader):
            inputs, targets = [b.to(device) for b in batch]

            outputs = model(input_ids=inputs)
            logits = outputs.logits

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches +=1

            if batch_index + 1 %50 ==0:
                avg_loss = total_loss /num_batches
                print(f"Epoch {epoch} | Step {batch_index}" f"| Avg loss: {avg_loss}")
            avg_loss = total_loss / num_batches
            print(f"Average Loss: {avg_loss}")

        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved to {args.output_dir}")

if __name__ == "__main__":
    main()







if __name__ == "__main__":


