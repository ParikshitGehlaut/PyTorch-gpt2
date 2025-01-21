import torch
from torch.nn import functional as F 
import math
import tiktoken
import numpy as np
import os
import sys
from hellaswag import render_example, iterate_examples
from gpt2 import GPT, GPTConfig
    
# -----------------------------------------------------------------------------------------------
# run the training loop
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1
print(f"Using DDP: {ddp}")

if ddp:
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    try:
        init_process_group(backend=backend)
    except ValueError as e:
        print(f"Failed to initialize process group: {e}")
        sys.exit(1)

    ddp_rank = int(os.environ.get('RANK', 0))
    ddp_local_rank = int(os.environ.get('LOCAL_RANK', 0))
    ddp_world_rank = int(os.environ.get('WORLD_RANK', 1))
    device = f'cuda:{ddp_local_rank}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_rank = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"


def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        assert split in {'train', 'val'}
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        
        assert len(shards) > 0, f"no shards found in {split} split"
        if master_process:
            print(f"shards in split {split} is {len(shards)}")
        
        self.reset()
    
    def reset(self):
        # state, start at shard zero
        self.current_shard = 0 
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
                    
    def next_batch(self):
        B, T = self.B, self.T
        
        buf = self.tokens[self.current_position : self.current_position + B * T + 1].clone().detach().to(device)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        
        self.current_position += B * T * self.num_processes
        
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
            
        return x, y
    
def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm
    

total_batch_size = 524288 # 2**19
B = 32 # micro batch size
T = 1024 # sequence length
grad_accum_steps = total_batch_size // (B * T * ddp_world_rank) # 32,   if ddp_world_rank is more than 1 then grad_accum_steps is less than 32 

if master_process:
    print(f"total desired batch size is {total_batch_size}")
    print(f"=> calculated gradient accumulation steps {grad_accum_steps}")
    
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_rank, split="train")
val_loader =  DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_rank, split="val")

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig(vocab_size=50304))
model.to(device=device)
# model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 500
max_steps = 18000

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
optimizer = raw_model.configure_optimizers(learning_rate=6e-4, weight_decay = 0.1, device=device)
torch.set_float32_matmul_precision('high') # use TensorFloat32 datatype

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 5000 == 0 or last_step):
                    # optionally write model checkpoints
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item()
                    }
                    # you might also want to add optimizer.state_dict() and
                    # rng seeds etc., if you wanted to more exactly resume training
                    torch.save(checkpoint, checkpoint_path)
                    
    # once in a while evaluate hellaswag
    if step % 250 == 0 or last_step:
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_rank != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    t1 = time.time()
    dt = (t1 - t0)  # time difference is in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_rank
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step: {step+1}| lr: {lr:.6f} | loss: {loss_accum.item():.6f}| norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
    
    
if ddp:
    destroy_process_group()

# import sys
# sys.exit(0)

# num_return_sequences = 5
# max_length = 30

# tokens = enc.encode("Hello, I'm a language Model")
# tokens = torch.tensor(tokens, dtype=torch.long, device=device) # shape: (8, )
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # shape: (5, 8)

# x = tokens

# # sampling loop
# # B --> Batch_size
# # T --> sequence_length
# # generate, right now B = 5 and T = 8 
# # set seed to 42
# torch.manual_seed(42)

# if torch.cuda.is_available():
#     torch.cuda.manual_seed(42)

# while x.size(1) < max_length:
#     with torch.no_grad():
#         logits = model(x)
#         # take logits at the last position
#         logits = logits[:, -1, :] # (B, vocab_size)
#         probs = F. softmax(logits, dim=-1)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

#         ix = torch.multinomial(topk_probs, 1) # (B, 1)
#         # gather the corresponding indices
#         xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
#         # append to the sequence
#         x = torch.cat((x, xcol), dim=1)
    
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     print(">", enc.decode(tokens))