import torch
import gpt
import re

def generate_text_simple(model,idx,max_new_tokens,context_size):
    for _ in range(max_new_tokens):
        idx_cond=idx[:,-context_size: ]
        with torch.no_grad():
            logits=model(idx_cond)

        logits=logits[:,-1,:]
        probas=torch.softmax(logits,dim=-1)
        idx_next=torch.argmax(probas,dim=-1,keepdim=True)
        idx=torch.cat((idx,idx_next),dim=1)
    return idx
        

GPT_CONFIG_NICO={
    "vocab_size":50257,
    "context_length":256,
    "emb_dim":768,
    "n_heads":12,
    "n_layers":10,
    "drop_rate":0.1,
    "qkv_bias":False
}



raw_text=""
with open("pesanteurEtLaGrace.txt","r",encoding="utf-8") as f:
    current_text=f.read()
raw_text += current_text
with open("enracinement.txt","r",encoding="utf-8") as f:
    current_text=f.read()
raw_text += current_text



preprocessed=re.split(r'([,.:;?_!"()\`]|--|\s)',raw_text)
all_words=sorted(set(preprocessed))
vocab_size=len(all_words)

vocab = {token:indice for indice,token in enumerate(all_words)}
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>","<|unk|>"])
vocab = {token:indice for indice,token in enumerate(all_tokens)}

tokenizer = gpt.SimpleTokenizerV2(vocab)

model=gpt.GPTModel(GPT_CONFIG_NICO)
model.load_state_dict(torch.load("modelGPTFr.pth"))
model.eval()

start_context="Le chat mange la souris "
encoded=tokenizer.encode(start_context)
encoded_tensor=torch.tensor(encoded).unsqueeze(0)
out=generate_text_simple(model=model,idx=encoded_tensor,max_new_tokens=60,context_size=GPT_CONFIG_NICO["context_length"])
decoded_text=tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
