
#   python lab1_final.py --path HP1.txt --window 2 --embed_dim 100 --epochs 5 --lr 0.1 --batch_size 512

import argparse
import re
import string
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# CLI args
# ----------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--path", type=str, default="HP1.txt", help="Path to the input text file")
ap.add_argument("--window", type=int, default=2, help="Context window size on each side")
ap.add_argument("--embed_dim", type=int, default=100, help="Embedding dimension")
ap.add_argument("--epochs", type=int, default=5, help="Training epochs")
ap.add_argument("--lr", type=float, default=0.1, help="Learning rate")
ap.add_argument("--batch_size", type=int, default=512, help="Mini-batch size")
ap.add_argument("--save_npz", type=str, default="embeddings_lab1.npz", help="Output file for learned matrices")
args = ap.parse_args()


# --- Reading text file (with simple fallbacks) ---
try_paths = [
    args.path,
    r"C:\Users\fzm1209\Documents\harry_potter\harry_potter\HP1.txt",
    "/content/HP1.txt",
    "/mnt/data/HP1.txt"
]

content = None
for p in try_paths:
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        print(f"Loaded text from: {p}")
        break
    except FileNotFoundError:
        continue
if content is None:
    raise FileNotFoundError(f"Could not find text file in any of: {try_paths}")

# Reading a first 10 sentences (preview, not used for training)
sent = re.split(r'[.!?]', content)[:10]
para_content = ' '.join(s.strip() for s in sent if s.strip())
print("Preview (first ~10 sentences):")
print(para_content[:500], "...\n")

# Split the text by whitespace (preview)
split_preview = content.split()
print("Raw split preview:", split_preview[:20], "\n")

# remove punctuation and lowercase
translator = str.maketrans('', '', string.punctuation)
content = content.translate(translator).lower()
# strip "'s" remnants
clean = [word.replace("'s", '') for word in content.split() if word]

print(f"Tokens (clean) count: {len(clean)}")

# Extract unique words from the text and collect them in an array
unique = []
seen = set()
for word in clean:
    if word not in seen:
        unique.append(word)
        seen.add(word)
print(f"Unique vocab size: {len(unique)}")

# Map each word to a unique index
index = {word: i for i, word in enumerate(unique)}
print("Index sample:", list(index.items())[:10])

# Create dataset with words as inputs and context as labels (window = args.window)
def create_dataset(clean, index, window=2):
    data = []
    n = len(clean)
    for i, target_word in enumerate(clean):
        context_indices = list(range(max(0, i - window), i)) + list(range(i + 1, min(n, i + window + 1)))
        for j in context_indices:
            data.append((index[target_word], index[clean[j]]))
    return data

dataset = create_dataset(clean, index, window=args.window)
print("Dataset size:", len(dataset))
print("Dataset sample:", dataset[:10])

# kip-Gram network + training + inference

# 1) Dataset wrapper 
class SkipGramPairsDataset(Dataset):
    def __init__(self, pairs: List[Tuple[int, int]]):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        c, ctx = self.pairs[idx]
        return torch.tensor(c, dtype=torch.long), torch.tensor(ctx, dtype=torch.long)

# Model: Embedding (|V| x d) + Linear(d -> |V|)
class SkipGramSoftmax(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)          
        self.out_linear = nn.Linear(embed_dim, vocab_size, bias=False) 
       
        nn.init.normal_(self.in_embed.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.out_linear.weight, mean=0.0, std=0.05)

    def forward(self, center_idxs):
        h = self.in_embed(center_idxs)     # (B, d)
        return self.out_linear(h)          # (B, |V|)

    # Inference function 
    @torch.no_grad()
    def one_hot_to_embedding(self, one_hot_batch):
        idxs = one_hot_batch.argmax(dim=-1)  
        return self.in_embed(idxs)           

# Training loop
def train(model, dataset, epochs=5, batch_size=512, lr=0.1, print_every=200, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=lr)

    step, running = 0, 0.0
    for ep in range(1, epochs + 1):
        for center, ctx in loader:
            center, ctx = center.to(device), ctx.to(device)
            logits = model(center)            # (B, |V|)
            loss = criterion(logits, ctx)     # context index as target

            optim.zero_grad()
            loss.backward()
            optim.step()

            running += loss.item()
            step += 1
            if step % print_every == 0:
                print(f"[Epoch {ep}] Step {step} | Loss={running/print_every:.4f}")
                running = 0.0

# Hook everything up and run
vocab_size = len(index)
torch_dataset = SkipGramPairsDataset(dataset)
model = SkipGramSoftmax(vocab_size=vocab_size, embed_dim=args.embed_dim)

print(f"Training: vocab={vocab_size} pairs={len(torch_dataset)} embed_dim={args.embed_dim}")
train(model, torch_dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, print_every=200)

# Save learned matrices 
in_W  = model.in_embed.weight.detach().cpu().numpy()    # central-word matrix V
out_W = model.out_linear.weight.detach().cpu().numpy()  # context-word matrix U
np.savez(args.save_npz, in_embed=in_W, out_embed=out_W, vocab_size=vocab_size, embed_dim=args.embed_dim)
print(f"Saved embeddings to {args.save_npz}")

# inference: one-hot -> embedding for vocab[0]
one_hot = torch.zeros((1, vocab_size), dtype=torch.float32)
one_hot[0, 0] = 1.0
emb = model.one_hot_to_embedding(one_hot)
print("Example embedding shape (for vocab[0]):", tuple(emb.shape))
print("Done.")
