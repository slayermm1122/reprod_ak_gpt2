import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000 #increase max iters because we learning rate is lower 
eval_interval = 500
learning_rate = 3e-4 #decrease learning rate b/c self attention can't tolerate very high learning rates
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.key = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.query = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.value = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # lower triangular mask
        self.proj = nn.Linear(num_heads * head_size, n_embd)  # projection back to n_embd
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # Compute key, query, value for all heads in parallel
        k = self.key(x)   # (B, T, num_heads * head_size)
        q = self.query(x) # (B, T, num_heads * head_size)
        v = self.value(x) # (B, T, num_heads * head_size)

        # Reshape to (B, num_heads, T, head_size) for multi-head attention
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, T, head_size)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, T, head_size)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, T, head_size)

        # Scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5) # (B, num_heads, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # apply mask (B, num_heads, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, num_heads, T, T)
        wei = self.dropout(wei)

        # Weighted sum over the values
        out = wei @ v  # (B, num_heads, T, head_size)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_size) # (B, T, num_heads * head_size)

        # Final linear projection back to n_embd
        out = self.proj(out)  # (B, T, n_embd)
        out = self.dropout(out) # (B, T, n_embd)
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention layer with LayerNorm, dropout, and residual connection
        x_residual = self.ln1(x)
        x = x + self.dropout(self.sa(x_residual))

        # FeedForward layer with LayerNorm, dropout, and residual connection
        x_residual = self.ln2(x)
        x = x + self.dropout(self.ffwd(x_residual))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #n_embd is the number of embedding dimensions
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) #final layer norm
        self.lm_head= nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) #this is embed C = n_embd.
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb #broadcasting in action. (B,T,C) + (1,T,C) -> (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print("Number of parameters:")
print(sum(p.numel() for p in m.parameters())/1e6, 'Mn parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# generate from the model before training
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("First generation before training:")
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

start_time = time.time()
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"elapsed time: {time.time() - start_time:.1f}s")
        print("="*50)

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("Generation after training:")
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
