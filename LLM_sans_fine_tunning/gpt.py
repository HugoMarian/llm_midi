import torch
import torch.nn as nn
from torch.utils.data import Dataset
import re


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        B, T, _ = x.shape

        q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(2, 3)
        scores = scores.masked_fill(self.mask[:T, :T].bool(), -1e9)
        attn = torch.softmax(scores / (self.head_dim ** 0.5), dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            cfg["emb_dim"], cfg["emb_dim"],
            cfg["context_length"], cfg["drop_rate"],
            cfg["n_heads"], cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.ln1 = LayerNorm(cfg["emb_dim"])
        self.ln2 = LayerNorm(cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = x + self.drop(self.att(self.ln1(x)))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

        self.blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.ln_f = LayerNorm(cfg["emb_dim"])
        self.head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)


class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.x, self.y = [], []
        ids = tokenizer.encode(text)

        for i in range(0, len(ids) - max_length + 1, stride):
            self.x.append(torch.tensor(ids[i:i+max_length]))
            self.y.append(torch.tensor(ids[i+1:i+max_length+1]))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.stoi = vocab
        self.itos = {i: s for s, i in vocab.items()}

    @staticmethod
    def tokenize(text):
        parts = re.split(r'([,.:;?_!"()\`]|--|\s)', text)
        return [p.strip() for p in parts if p and p.strip()]

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.stoi.get(t, self.stoi["<|unk|>"]) for t in tokens]

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids)
