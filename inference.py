import torch
from torch.nn import functional as F
import tiktoken
import warnings
from gpt2 import GPT, GPTConfig

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
print(f"Using device {device}")

model = GPT(GPTConfig(vocab_size=50304))
weights_path = "./log/model_10000.pt"

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

model_weights = state_dict["model"]
model.load_state_dict(model_weights, strict=True)
model.to(device=device)


num_return_sequences = 5
max_length = 100

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Tell me a joke?")
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
        logits, loss = model(x)
        # take logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)
    
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    print(">", enc.decode(tokens))