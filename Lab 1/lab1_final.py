# hyper-parameter to be tweaked to find optimal network
import argparse
import os
import re
import string
from collections import Counter, OrderedDict
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--folder", type=str, default=r"C:\Users\fzm1209\Documents\harry_potter\harry_potter",
                help="Folder with HP text files (HP1..HP7)")
ap.add_argument("--path", type=str, default="", help="Path to a single text file")
ap.add_argument("--window", type=int, default=2, help="Max context window size (dynamic)")
ap.add_argument("--embed_dim", type=int, default=100, help="Embedding dimension")
ap.add_argument("--epochs", type=int, default=10, help="Training epochs")
ap.add_argument("--lr", type=float, default=0.003, help="Learning rate")
ap.add_argument("--batch_size", type=int, default=1024, help="Mini-batch size")
ap.add_argument("--neg_k", type=int, default=10, help="Number of negative samples")
ap.add_argument("--min_count", type=int, default=5, help="Min frequency to keep a word")
ap.add_argument("--subsample_t", type=float, default=1e-5, help="Subsampling threshold")
ap.add_argument("--save_npz", type=str, default="embeddings_lab1_opt.npz", help="Output file for learned matrices")
args = ap.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read text from either all files in a folder or a single file
def read_corpus():
    texts = []
    if args.folder:
        for fname in sorted(os.listdir(args.folder)):
            if fname.lower().endswith(".txt"):
                with open(os.path.join(args.folder, fname), "r", encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read())
    elif args.path:
        with open(args.path, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    else:
        raise ValueError("Provide either --folder or --path")
    return " ".join(texts)

# Remove punctuation, lowercase text, and split into tokens
def clean_tokens(text):
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator).lower()
    return [w.replace("'s", '') for w in text.split() if w]

# Remove rare words (min_count) and subsample frequent words
def prune_and_subsample(tokens, min_count, subsample_t):
    freq = Counter(tokens)
    tokens = [w for w in tokens if freq[w] >= min_count]
    total = len(tokens)
    def keep(w):
        p = freq[w] / total
        prob = (np.sqrt(p / subsample_t) + 1) * (subsample_t / p)
        return np.random.rand() < prob
    return [w for w in tokens if keep(w)], freq

# Build vocabulary mapping from word to index and index to word
def build_vocab(tokens):
    seen = OrderedDict()
    for w in tokens:
        if w not in seen:
            seen[w] = len(seen)
    return dict(seen), {i: w for w, i in seen.items()}

# Generate (center, context) pairs using a dynamic context window
def build_pairs(tokens, word2idx, max_window):
    ids = [word2idx[w] for w in tokens if w in word2idx]
    pairs = []
    for i in range(len(ids)):
        w = np.random.randint(1, max_window + 1)
        for j in range(max(0, i - w), i):
            pairs.append((ids[i], ids[j]))
        for j in range(i + 1, min(len(ids), i + w + 1)):
            pairs.append((ids[i], ids[j]))
    return pairs

class PairsDataset(Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): c, ctx = self.pairs[i]; return torch.tensor(c), torch.tensor(ctx)

class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.in_embed.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.out_embed.weight, mean=0.0, std=0.05)
    def forward_pos(self, center_idx, pos_idx):
        v = self.in_embed(center_idx)
        u_pos = self.out_embed(pos_idx)
        return torch.sum(v * u_pos, dim=1), v
    def forward_neg(self, v, neg_idx):
        u_neg = self.out_embed(neg_idx)
        return torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)
    @torch.no_grad()
    def one_hot_to_embedding(self, one_hot_batch):
        idxs = one_hot_batch.argmax(dim=-1)
        return self.in_embed(idxs)

# Train Skip-Gram with Negative Sampling
def train_ns(model, loader, epochs, k_neg, noise_dist):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
    step, running = 0, 0.0
    for ep in range(1, epochs + 1):
        for center, pos in loader:
            center, pos = center.to(DEVICE), pos.to(DEVICE)
            score_pos, v = model.forward_pos(center, pos)
            loss_pos = F.binary_cross_entropy_with_logits(score_pos, torch.ones_like(score_pos))
            neg = torch.from_numpy(np.random.choice(len(noise_dist), size=(center.size(0), k_neg), p=noise_dist)).to(DEVICE)
            score_neg = model.forward_neg(v, neg)
            loss_neg = F.binary_cross_entropy_with_logits(-score_neg, torch.ones_like(score_neg))
            loss = loss_pos + loss_neg
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); step += 1
            if step % 200 == 0:
                print(f"[Epoch {ep}] Step {step} | Loss={running/200:.4f}")
                running = 0.0
        sched.step()

text = read_corpus()
tokens = clean_tokens(text)
tokens, freq = prune_and_subsample(tokens, args.min_count, args.subsample_t)
word2idx, idx2word = build_vocab(tokens)
pairs = build_pairs(tokens, word2idx, args.window)
dataset = PairsDataset(pairs)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

counts = np.array([freq[w] for w in word2idx], dtype=np.float64)
noise_dist = counts ** 0.75; noise_dist /= noise_dist.sum()

model = SkipGramNS(len(word2idx), args.embed_dim).to(DEVICE)
print(f"Training: vocab={len(word2idx)}, pairs={len(dataset)}, embed_dim={args.embed_dim}")
train_ns(model, loader, args.epochs, args.neg_k, noise_dist)

in_W = model.in_embed.weight.detach().cpu().numpy()
out_W = model.out_embed.weight.detach().cpu().numpy()
np.savez(args.save_npz, in_embed=in_W, out_embed=out_W, vocab_size=len(word2idx), embed_dim=args.embed_dim)
print(f"Saved embeddings to {args.save_npz}")

one_hot = torch.zeros((1, len(word2idx)), device=DEVICE)
one_hot[0, 0] = 1.0
emb = model.one_hot_to_embedding(one_hot)
print("Example embedding shape:", tuple(emb.shape))
