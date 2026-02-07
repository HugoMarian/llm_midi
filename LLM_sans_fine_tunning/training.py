import json
import torch
import gpt
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

files = ["pesanteurEtLaGrace.txt", "enracinement.txt"]
raw_text = ""

for f in files:
    with open(f, "r", encoding="utf-8") as fh:
        raw_text += fh.read() + "\n"

tokens = gpt.SimpleTokenizerV2.tokenize(raw_text)
vocab = {t: i for i, t in enumerate(sorted(set(tokens)))}
vocab["<|endoftext|>"] = len(vocab)
vocab["<|unk|>"] = len(vocab)

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)

tokenizer = gpt.SimpleTokenizerV2(vocab)

CFG = {
    "vocab_size": len(vocab),
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 10,
    "drop_rate": 0.1,
    "qkv_bias": False
}

model = gpt.GPTModel(CFG).to(device)

dataset = gpt.GPTDatasetV1(raw_text, tokenizer, max_length=100, stride=50)
loader = DataLoader(dataset, batch_size=10, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

for epoch in range(10):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0,1), y.flatten()
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch+1} | loss {loss.item():.4f}")
    torch.save(model.state_dict(), "modelGPTFr.pth")
