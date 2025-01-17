from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F 
import math
import inspect 

# -----------------------------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768    
    
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        # regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embed)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        
    def forward(self, x):
        x = self.c_proj(self.gelu(self.c_fc(x)))
        return x

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x1 = self.ln_1(x)
        x2 = self.attn(x1)
        x = x + x2

        x1 = self.ln_2(x)
        x2 = self.mlp(x1)
        x = x + x2

        return x

    
    
class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config 
        
        self.transformer = nn.ModuleDict(dict(
            # weight of token embedding
            wte = nn.Embedding(config.vocab_size, config.n_embed), # (50384, 384)
            # weight of position embedding
            wpe = nn.Embedding(config.block_size, config.n_embed), # (block_size, 384)
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed),
        ))
        
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # apply init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets=None):
        # shape of idx is: (B, T)
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embed)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embed)
        x = tok_emb + pos_emb
        
        # shape of x is (B, T, n_embed)
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        # (B, T, n_embed) --> (B, T, vocab_size)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def configure_optimizers(self, learning_rate, weight_decay, device):
        # pn --> parameter name
        # p --> parameter itself
        param_dict = {pn : p for pn, p in self.named_parameters()}
        param_dict = {pn : p for pn, p in param_dict.items() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay' : weight_decay},
            {'params': nodecay_params, 'weight_decay' : 0.0}
        ]
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
         
        
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embed are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# -----------------------------------------------------------------------------------------------
import tiktoken

num_return_sequences = 5
max_length = 30

ALLOW_CUDA = True
ALLOW_MPS = True

device = "cpu"
if torch.cuda.is_available() and ALLOW_CUDA:
    device = "cuda"
elif torch.backends.mps.is_available() and ALLOW_MPS:
    device = "mps"
    
print(f"Using device {device}")
    
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        with open('input.txt', 'r') as f:
            data = f.read()
            
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(data)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")
        
        self.current_position = 0
        
    def next_batch(self):
        B, T = self.B, self.T
        
        buf = self.tokens[self.current_position : self.current_position + B * T + 1].clone().detach().to(device)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        
        self.current_position += B*T
        
        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_position = 0
            
        return x, y
    
    
total_batch_size = 524288 # 2**19
B = 16 # micro batch size
T = 1024 # sequence length
grad_accum_steps = total_batch_size // (B * T) # 32
print(f"total desired batch size is {total_batch_size}")
print(f"=> calculated gradient accumulation steps {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T)

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig(vocab_size=50304))
model.to(device=device)
# Use torch.compile() with cpu or cuda gpu
model = torch.compile(model)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 200

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1)/warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr 
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos (math.pi * decay_ratio)) # coeff starts at 1 and goes to ® 
    return min_lr + coeff * (max_lr - min_lr)

import time 
optimizer = model.configure_optimizers(learning_rate=6e-4, weight_decay = 0.1, device=device)
torch.set_float32_matmul_precision('high') # use TensorFloat32 datatype

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)  # time difference is in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps)/ (t1 - t0)
    if (step+1)%10 == 0:
        print(f"step: {step+1}| lr: {lr:.6f} | loss: {loss_accum.item():.6f}| norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    

import sys
sys.exit(0)

tokens = enc.encode("Hello, I'm a language Model")
tokens = torch.tensor(tokens, dtype=torch.long, device=device) # shape: (8, )
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # shape: (5, 8)

x = tokens

# sampling loop
# B --> Batch_size
# T --> sequence_length
# generate, right now B = 5 and T = 8 
# set seed to 42
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        # take logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        probs = F. softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)
    
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    print(">", enc.decode(tokens))