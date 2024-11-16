import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
import transformers
import tiktoken
import sys
import time

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes token + 1 <endoftext> token
    n_head: int = 12 # number of attention heads
    n_layer: int = 12 # number of layers
    n_embd: int = 768 # embedding dimensionality

class DataLoader():
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoc = {len(self.tokens) // (B*T)} batches")

        #state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x = buf[:-1].view(B, T) # input 
        y = buf[1:].view(B, T) # target
        self.current_position += B*T
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        return x, y

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones_like(torch.zeros(config.block_size, config.block_size)))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # GPT-2 uses GELU instead of ReLU
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) 

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) #Shape T
        tok_emb = self.transformer.wte(idx) #Shape B, T, C
        pos_emb = self.transformer.wpe(pos) #Shape T, C
        x = tok_emb + pos_emb
        
        #forward the blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) #note: the final layer norm is not included in the final attn block
        logits = self.lm_head(x) # note: using list [-1] to preserve the time dim
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) #(B*T, vocab_size)
        # logits size B, T, vocab_size
        
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained model weights from Hugging Face"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head, n_embd are determined by model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), #124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), #355M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), #774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), #1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 #gpt-2 vocab_size
        config_args['block_size'] = 1024 #gpt-2 block_size
        # create a from-scratch initialized model
        model = GPT(GPTConfig(**config_args))
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask/buffer, not a param

        # init a huggingface model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # discard this mask/buffer, not a param
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # discard this mask/buffer, not a param
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatch between model_hf: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special handling for the weights in the attn module
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad(): 
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad(): 
                    sd[k].copy_(sd_hf[k])
        return model

#--------------------------------
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")

model = GPT(GPTConfig())
model.to(device)

#--------------------------------
enc = tiktoken.get_encoding("gpt2")
with open("input.txt", "r") as f:
    text = f.read()

tokens = enc.encode(text)
#print(f"Number of tokens: {len(tokens)}")

B, T =4,32
buf = torch.tensor(tokens[:B*T+1])
buf = buf.to(device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

logits, loss = model(x, y)
print(loss)

#sys.exit()

#训练
#train_loader = DataLoader(B=4, T=32)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(10):
    start_time = time.time()  # 记录开始时间
    #x, y = train_loader.next_batch()
    #x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算耗时
    print(f"第 {i} 次迭代，损失为: {loss.item():.4f}，耗时: {elapsed_time:.4f} 秒")


sys.exit()

#--------------------------------
num_return_sequences = 5
max_length = 30

#prefix tokens
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I want to learn python, so")
tokens = torch.tensor(tokens, dtype=torch.long) #(8,) if use gpt2 encoding
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) #(num_return_sequences, 8)
x = tokens.to(device)

#generate! right now x is (B, T) where B = 5, T = 8
#set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    logits = model(x) #(B, T, vocab_size)
    logits = logits[:, -1, :] #(B, vocab_size)
    probs = F.softmax(logits, dim=-1) #(B, vocab_size)
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    ix = torch.multinomial(topk_probs, 1) #(B, 1)
    xcol = torch.gather(topk_indices, -1, ix) #(B, 1)
    x = torch.cat((x, xcol), dim=1) #(B, T+1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)