# PyTorch-gpt2

This repository contains the implementation, training and inference script for the GPT-2 124M model using PyTorch. The training script uses distributed training using DistributedDataParallel (DDP) to utilise hardware completely while training on more one Nvidia GPU.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [inference](#inference)

---


## Requirements

Before getting started, ensure you have the following:

- Python >= 3.10
- PyTorch >= 2.0
- Additional dependencies listed in `requirement.txt`

Install the required packages with:
```bash
pip install -r requirement.txt
```

---

## Installation

Clone this repository to your local machine:
```bash
git clone [https://github.com/ParikshitGehlaut/PyTorch-gpt2.git](https://github.com/ParikshitGehlaut/PyTorch-gpt2.git)
cd PyTorch-gpt2
conda activate myenv
```

Download Fineweb_edu dataset:
```bash
python fineweb.py
```

Download Hellaswag:
```bash
python hellaswag.py
```

---

## Usage

### Training
1. **Using DDP**:
   replace <device_count> with number of GPUs like 2 or 4.
   ```bash
   torchrun --standalone --nproc_per_node=<device_count> train.py
   ```

2. **Not Using DDP**:
   ```bash
   CUDA_VISIBLE_dEVICES=0 python train.py
   ```

### Inference

   **Run the Inference Script**:
   ```bash
   python inference.py
   ```

---

Feel free to raise issues or contribute to this repository to improve its functionality.
