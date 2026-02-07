import json
import torch
import gpt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
vocab = {k: int(v) for k, v in vocab.items()}

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
model.load_state_dict(torch.load("modelGPTFr.pth", map_location=device))
model.eval()


def generate(model, idx, steps):
    for _ in range(steps):
        idx_cond = idx[:, -CFG["context_length"]:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


prompt = "Abel POSITION_200"
ids = tokenizer.encode(prompt)
idx = torch.tensor(ids, device=device).unsqueeze(0)

out = generate(model, idx, 200)[0].tolist()
tokens = [tokenizer.itos[i] for i in out]

with open("test_Abel_pos_200.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(tokens))
