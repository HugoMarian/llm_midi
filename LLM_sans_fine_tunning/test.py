import torch
import gpt
import re
import os
from random import sample

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


raw_text = ""
i = 0
num_texts= 5
d = sample(os.listdir("../training"), num_texts)
for e in d:
        current_text = e.replace(".txt", "")
        try:
            with open("../training/"+current_text+".txt","r",encoding="utf-8") as f:
                current_text+=f.read()
                f.close()
            raw_text += current_text
        except:
            continue



preprocessed=re.split(r'([,.:;?_!"()\`]|--|\s)',raw_text)
all_words=sorted(set(preprocessed))
vocab_size=len(all_words)

vocab = {token:indice for indice,token in enumerate(all_words)}
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>","<|unk|>"])
vocab = {token:indice for indice,token in enumerate(all_tokens)}

tokenizer = gpt.SimpleTokenizerV2(vocab)

model=gpt.GPTModel(GPT_CONFIG_NICO)
model.load_state_dict(torch.load("modelGPTmidiAll.pth"))
model.eval()

start_context="Morales POSITION_200"
encoded=tokenizer.encode(start_context)
encoded_tensor=torch.tensor(encoded).unsqueeze(0)
out=generate_text_simple(model=model,idx=encoded_tensor,max_new_tokens=200,context_size=GPT_CONFIG_NICO["context_length"])
decoded_text=tokenizer.decode(out.squeeze(0).tolist())

with open("../tests/test_Morales.txt","w",encoding="utf-8") as f:
    for ligne in decoded_text.replace(" _ ", "_").split(" "):
        f.write(ligne+"\n")
    f.close()
