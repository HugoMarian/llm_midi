import re
import torch
import gpt
import subprocess
import os
from random import sample

from torch.utils.data import Dataset,DataLoader

def calc_loss_batch(input_batch,target_batch,model,device):
    input_batch=input_batch.to(device)
    target_batch=target_batch.to(device)
    logits=model(input_batch)
    loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten())
    return loss

def train_model_simple(model,train_loader,val_loader,optimizer,device,num_epochs,eval_freq,eval_iter,start_context,tokenizer):
    train_losses,val_losses,track_tokens_seen=[],[],[]
    tokens_seen,global_step=0,-1

    for epoch in range(num_epochs):
        print(f'device: {device}')
        print(f'epoch: {epoch}')
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss=calc_loss_batch(input_batch,target_batch,model,device)
            loss.backward()
            optimizer.step()
            tokens_seen+=input_batch.numel()
            global_step+=1
#            print(loss)
        #    if global_step % eval_freq == 0:
        #        train_loss,val_loss=evaluate_model(model,train_loader,val_loader,device,eval_iter)
        #        train_losses.append(train_loss)
        #        val_losses.append(val_loss)
        #        track_tokens_seen.append(tokens_seen)

        # generate_and_print_sample(model,tokenizer,device_start_context)
        torch.save(model.state_dict(),"modelGPTmidiAll.pth")

    return train_losses,val_losses,track_tokens_seen


#lecture du fichier
# with open("../fichiers/exempleFichierToken.txt","r",encoding="utf-8") as f:
#    raw_text=f.read()
#    f.close()
    
# # tokenizer
# preprocessed=re.split(r'([,.:;?!"()\`]|--|\s)',raw_text)

# all_words=sorted(set(preprocessed))
# vocab_size=len(all_words)

# vocab = {token:indice for indice,token in enumerate(all_words)}
# all_tokens = sorted(list(set(preprocessed)))
# all_tokens.extend(["<|endoftext|>","<|unk|>"])
# vocab = {token:indice for indice,token in enumerate(all_tokens)}
# tokenizer = gpt.SimpleTokenizerV2(vocab)

# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
GPT_CONFIG_NICO={
    "vocab_size":50257,
    "context_length":256,
    "emb_dim":768,
    "n_heads":12,
    "n_layers":10,
    "drop_rate":0.1,
    "qkv_bias":False
}

#relearn=True
relearn=False
model=gpt.GPTModel(GPT_CONFIG_NICO)

model.to(device)
if relearn==True:
    model.load_state_dict(torch.load("modelGPTmidiAll.pth", map_location=device))
    model.to(device)

raw_text=""

# Transformation .mid -> tokens
print("Tokenization...")
num_texts= 10
d = sample(os.listdir("../GrandMidiPiano"), num_texts)
for e in d:
    current_text = e.replace(".mid", "")
    subprocess.run(["python", "../fichiers/midi2tokens.py", current_text+".mid"])
    try:
        with open("../training/"+current_text+".txt","r",encoding="utf-8") as f:
            current_text+=f.read()
            f.close()
        raw_text += current_text
    except:
        continue
print("Done")

# with open("../fichiers/exempleFichierToken.txt","r",encoding="utf-8") as f:
#    raw_text=f.read()
#    f.close()
    
# # tokenizer
# with open("../fichiers/exempleFichierToken.txt","r",encoding="utf-8") as f:
#     current_text=f.read()
# raw_text += current_text
# with open("../fichiers/test.txt","r",encoding="utf-8") as f:
#     current_text=f.read()
# raw_text += current_text
    
preprocessed=re.split(r'([,.:;?_!"()\`]|--|\s)',raw_text)
all_words=sorted(set(preprocessed))
vocab_size=len(all_words)

vocab = {token:indice for indice,token in enumerate(all_words)}
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>","<|unk|>"])
vocab = {token:indice for indice,token in enumerate(all_tokens)}

tokenizer = gpt.SimpleTokenizerV2(vocab)

train_ds=gpt.GPTDatasetV1(raw_text,tokenizer,max_length=100,stride=50)

train_loader=DataLoader(dataset=train_ds,batch_size=10,shuffle=True,num_workers=0)
#print(len(train_loader))
val_loader=DataLoader(dataset=train_ds,batch_size=10,shuffle=True,num_workers=0)
optimizer=torch.optim.AdamW(model.parameters(),lr=0.0004,weight_decay=0.1)
num_epochs=10
eval_freq=5
eval_iter=5
start_context=""

train_model_simple(model,train_loader,val_loader,optimizer,device,num_epochs,eval_freq,eval_iter,start_context,tokenizer)
torch.save(model.state_dict(),"modelGPTmidiAll.pth")
